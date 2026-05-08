"""Build deterministic candidates by replacing parser-proven monolithic sections."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli

from tac.frontier_archive_layout import inspect_frontier_archive_layout


class MonolithicPacketCandidateError(ValueError):
    """Raised when a packet replacement would violate custody or grammar."""


CANONICAL_DISPATCH_CLAIMS_PATH = ".omx/state/active_lane_dispatch_claims.md"
RUNTIME_READY_FIELDS = (
    "ready_for_exact_eval_runtime",
)
RUNTIME_ARCHIVE_SHA_FIELDS = (
    "candidate_archive_sha256",
    "archive_sha256",
    "output_archive_sha256",
    "candidate_sha256",
)
RUNTIME_MEMBER_SHA_FIELDS = (
    "rebuilt_member_sha256",
    "new_member_sha256",
    "member_sha256",
    "payload_sha256",
    "output_member_sha256",
)
TERMINAL_CLAIM_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)
RUNTIME_CONSUMPTION_PROOF_SCHEMA = "tac_runtime_consumption_proof_v1"
ACTIVE_LANE_CLAIM_JSON_SCHEMA = "tac_active_lane_claim_json_v1"
PR106_BROTLI_SECTION_NAMES = frozenset(
    {"decoder_packed_brotli", "latents_and_sidecar_brotli"}
)


@dataclass(frozen=True)
class ReplacementSection:
    """Section replacement request for a parser-proven monolithic packet."""

    section_name: str
    replacement_path: Path
    expected_old_sha256: str | None = None
    expected_old_bytes: int | None = None
    expected_new_sha256: str | None = None
    expected_new_bytes: int | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_monolithic_packet_candidate(
    *,
    source_archive: Path,
    output_archive: Path,
    candidate_id: str,
    replacements: list[ReplacementSection],
    expected_source_archive_sha256: str | None = None,
    expected_source_archive_bytes: int | None = None,
    manifest_output: Path | None = None,
    runtime_parity: Mapping[str, Any] | None = None,
    lane_claim: Mapping[str, Any] | None = None,
    dispatch_lane_id: str | None = None,
    dispatch_instance_job_id: str | None = None,
) -> dict[str, Any]:
    """Replace logical sections inside a single-member HNeRV packet.

    The emitted candidate is archive-construction evidence only. It never
    claims score, promotion, rank, or kill eligibility. Dispatch readiness is
    fail-closed behind runtime-consumption proof and an active lane claim;
    exact CUDA auth eval remains the promotion gate.
    """

    if not candidate_id:
        raise MonolithicPacketCandidateError("candidate_id is required")
    if not replacements:
        raise MonolithicPacketCandidateError("at least one replacement is required")

    source_archive = Path(source_archive)
    output_archive = Path(output_archive)
    source_bytes = source_archive.stat().st_size
    source_sha = sha256_file(source_archive)
    if expected_source_archive_bytes is not None and source_bytes != expected_source_archive_bytes:
        raise MonolithicPacketCandidateError(
            f"source archive bytes mismatch: {source_bytes} != {expected_source_archive_bytes}"
        )
    if expected_source_archive_sha256 and source_sha.lower() != expected_source_archive_sha256.lower():
        raise MonolithicPacketCandidateError(
            "source archive sha256 mismatch: "
            f"{source_sha} != {expected_source_archive_sha256}"
        )

    layout = inspect_frontier_archive_layout(source_archive)
    physical = layout.get("physical_layout")
    logical = layout.get("logical_layout")
    if not isinstance(physical, dict) or not physical.get("single_member_monolithic_packet"):
        raise MonolithicPacketCandidateError("source archive is not a single-member monolithic packet")
    if not isinstance(logical, dict):
        raise MonolithicPacketCandidateError("source archive has no parser-proven logical layout")
    sections = logical.get("sections")
    if not isinstance(sections, list):
        raise MonolithicPacketCandidateError("logical layout has no sections")

    member = _single_member_payload(source_archive)
    section_by_name = {
        str(section.get("name")): section
        for section in sections
        if isinstance(section, dict) and section.get("name")
    }
    replacement_by_name: dict[str, ReplacementSection] = {}
    for replacement in replacements:
        if replacement.section_name in replacement_by_name:
            raise MonolithicPacketCandidateError(
                f"duplicate replacement section: {replacement.section_name}"
            )
        if replacement.section_name not in section_by_name:
            raise MonolithicPacketCandidateError(
                f"unknown parser-proven section: {replacement.section_name}"
            )
        replacement_by_name[replacement.section_name] = replacement

    grammar = str(logical.get("grammar"))
    if grammar == "pr101_fixed_offset_hnerv_microcodec":
        _validate_pr101_replacements(replacement_by_name, section_by_name)
    elif grammar != "pr106_ff_packed_hnerv":
        raise MonolithicPacketCandidateError(f"unsupported monolithic grammar: {grammar}")
    elif "ff_header" in replacement_by_name:
        raise MonolithicPacketCandidateError("ff_header is derived and cannot be user-replaced")

    _validate_sections(member, sections, offset_key="offset", len_key="len", sha_key="sha256")
    if grammar == "pr106_ff_packed_hnerv":
        _validate_pr106_brotli_sections(
            member,
            sections,
            offset_key="offset",
            len_key="len",
            packet_label="source",
        )

    new_sections: list[dict[str, Any]] = []
    new_payload_parts: list[bytes] = []
    cursor = 0
    replacement_manifests: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            raise MonolithicPacketCandidateError("section entry is not an object")
        name = str(section["name"])
        old_offset = int(section["offset"])
        old_len = int(section["len"])
        old_data = member[old_offset:old_offset + old_len]
        old_sha = sha256_bytes(old_data)
        if old_sha != section.get("sha256"):
            raise MonolithicPacketCandidateError(
                f"source section sha mismatch for {name}: {old_sha} != {section.get('sha256')}"
            )
        replacement = replacement_by_name.get(name)
        if replacement is None:
            new_data = old_data
            changed = False
        else:
            new_data = replacement.replacement_path.read_bytes()
            if new_data == old_data:
                raise MonolithicPacketCandidateError(
                    f"no-op replacement rejected for {name}; payload bytes are unchanged"
                )
            changed = True
            _check_expected_section(replacement, old_data=old_data, new_data=new_data)
            replacement_manifests.append(
                {
                    "section_name": name,
                    "old_offset": old_offset,
                    "old_bytes": old_len,
                    "old_sha256": old_sha,
                    "new_bytes": len(new_data),
                    "new_sha256": sha256_bytes(new_data),
                    "section_byte_delta": len(new_data) - old_len,
                    "replacement_path": str(replacement.replacement_path),
                }
            )
        new_payload_parts.append(new_data)
        new_sections.append(
            {
                "name": name,
                "role": section.get("role"),
                "old_offset": old_offset,
                "new_offset": cursor,
                "old_len": old_len,
                "new_len": len(new_data),
                "old_sha256": old_sha,
                "new_sha256": sha256_bytes(new_data),
                "changed": changed,
            }
        )
        cursor += len(new_data)

    new_member = b"".join(new_payload_parts)
    if grammar == "pr106_ff_packed_hnerv":
        new_member = _rewrite_pr106_header(new_member, new_sections)
        new_sections = _refresh_pr106_header_section(new_member, new_sections)
    _validate_sections(
        new_member,
        new_sections,
        offset_key="new_offset",
        len_key="new_len",
        sha_key="new_sha256",
    )
    if grammar == "pr106_ff_packed_hnerv":
        _validate_pr106_brotli_sections(
            new_member,
            new_sections,
            offset_key="new_offset",
            len_key="new_len",
            packet_label="candidate",
        )

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    member_name = str(logical["single_member_name"])
    _write_single_member_zip(output_archive, member_name=member_name, payload=new_member)
    candidate_bytes = output_archive.stat().st_size
    candidate_sha = sha256_file(output_archive)
    new_member_sha = sha256_bytes(new_member)
    runtime_parity_summary, runtime_blockers = _runtime_parity_summary(
        runtime_parity,
        candidate_archive_sha256=candidate_sha,
        new_member_sha256=new_member_sha,
        replacement_manifests=replacement_manifests,
    )
    lane_claim_summary, lane_claim_blockers = _lane_claim_summary(
        lane_claim,
        expected_lane_id=dispatch_lane_id,
        expected_instance_job_id=dispatch_instance_job_id,
    )
    dispatch_blockers = [*runtime_blockers, *lane_claim_blockers]
    promotion_blockers = ["contest_cuda_auth_eval_missing"]

    manifest = {
        "schema": "tac_monolithic_packet_candidate_v1",
        "candidate_id": candidate_id,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": not dispatch_blockers,
        "evidence_grade": "empirical_archive_construction_no_score",
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_bytes,
            "sha256": source_sha,
        },
        "candidate_archive": {
            "path": str(output_archive),
            "bytes": candidate_bytes,
            "sha256": candidate_sha,
            "archive_byte_delta": candidate_bytes - source_bytes,
        },
        "monolithic_layout": {
            "grammar": grammar,
            "member_name": member_name,
            "old_member_bytes": len(member),
            "old_member_sha256": sha256_bytes(member),
            "new_member_bytes": len(new_member),
            "new_member_sha256": new_member_sha,
            "member_byte_delta": len(new_member) - len(member),
            "sections": new_sections,
        },
        "replacements": replacement_manifests,
        "runtime_parity": runtime_parity_summary,
        "lane_claim": lane_claim_summary,
        "dispatch_blockers": dispatch_blockers,
        "promotion_blockers": promotion_blockers,
        "notes": [
            "This mutates parser-proven sections inside one monolithic ZIP member.",
            "No score, promotion, rank, or kill claim is allowed from this manifest alone.",
            "Dispatch readiness only means runtime-consumption proof plus an active Level-2 lane claim.",
            "Promotion remains blocked until exact CUDA auth eval on the exact archive lands.",
        ],
    }
    if manifest_output is not None:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def _single_member_payload(source_archive: Path) -> bytes:
    with zipfile.ZipFile(source_archive) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise MonolithicPacketCandidateError(
                f"expected exactly one ZIP member, found {len(infos)}"
            )
        _validate_zip_member_name(infos[0].filename)
        with zf.open(infos[0], "r") as f:
            return f.read()


def _validate_pr101_replacements(
    replacement_by_name: dict[str, ReplacementSection],
    section_by_name: dict[str, dict[str, Any]],
) -> None:
    for name, replacement in replacement_by_name.items():
        section = section_by_name[name]
        old_len = int(section["len"])
        new_len = replacement.replacement_path.stat().st_size
        if name != "sidecar_blob" and new_len != old_len:
            raise MonolithicPacketCandidateError(
                "PR101 fixed-offset grammar only permits equal-length replacement "
                f"for {name}; got {new_len} vs {old_len}"
            )


def _check_expected_section(
    replacement: ReplacementSection,
    *,
    old_data: bytes,
    new_data: bytes,
) -> None:
    old_sha = sha256_bytes(old_data)
    new_sha = sha256_bytes(new_data)
    if replacement.expected_old_sha256 and old_sha.lower() != replacement.expected_old_sha256.lower():
        raise MonolithicPacketCandidateError(
            "old section sha256 mismatch: "
            f"{old_sha} != {replacement.expected_old_sha256}"
        )
    if replacement.expected_old_bytes is not None and len(old_data) != replacement.expected_old_bytes:
        raise MonolithicPacketCandidateError(
            f"old section bytes mismatch: {len(old_data)} != {replacement.expected_old_bytes}"
        )
    if replacement.expected_new_sha256 and new_sha.lower() != replacement.expected_new_sha256.lower():
        raise MonolithicPacketCandidateError(
            "new section sha256 mismatch: "
            f"{new_sha} != {replacement.expected_new_sha256}"
        )
    if replacement.expected_new_bytes is not None and len(new_data) != replacement.expected_new_bytes:
        raise MonolithicPacketCandidateError(
            f"new section bytes mismatch: {len(new_data)} != {replacement.expected_new_bytes}"
        )


def _validate_sections(
    member: bytes,
    sections: list[Any],
    *,
    offset_key: str,
    len_key: str,
    sha_key: str,
) -> None:
    cursor = 0
    names: set[str] = set()
    for section in sections:
        if not isinstance(section, Mapping):
            raise MonolithicPacketCandidateError("section entry is not an object")
        name = section.get("name")
        if not isinstance(name, str) or not name:
            raise MonolithicPacketCandidateError("section name missing")
        if name in names:
            raise MonolithicPacketCandidateError(f"duplicate section name: {name}")
        names.add(name)
        offset = section.get(offset_key)
        length = section.get(len_key)
        if not isinstance(offset, int) or not isinstance(length, int):
            raise MonolithicPacketCandidateError(f"section {name} offset/len is not integer")
        if offset != cursor:
            raise MonolithicPacketCandidateError(
                f"section {name} is not contiguous: offset {offset} != {cursor}"
            )
        if length < 0:
            raise MonolithicPacketCandidateError(f"section {name} has negative length")
        end = offset + length
        if end > len(member):
            raise MonolithicPacketCandidateError(f"section {name} extends past member")
        expected_sha = section.get(sha_key)
        if expected_sha and sha256_bytes(member[offset:end]) != expected_sha:
            raise MonolithicPacketCandidateError(f"section {name} sha mismatch")
        cursor = end
    if cursor != len(member):
        raise MonolithicPacketCandidateError(
            f"sections do not cover member: final cursor {cursor} != {len(member)}"
        )


def _validate_pr106_brotli_sections(
    member: bytes,
    sections: list[Any],
    *,
    offset_key: str,
    len_key: str,
    packet_label: str,
) -> None:
    seen: set[str] = set()
    for section in sections:
        if not isinstance(section, Mapping):
            raise MonolithicPacketCandidateError("section entry is not an object")
        name = section.get("name")
        if name not in PR106_BROTLI_SECTION_NAMES:
            continue
        offset = int(section[offset_key])
        length = int(section[len_key])
        try:
            brotli.decompress(member[offset:offset + length])
        except brotli.error as exc:
            raise MonolithicPacketCandidateError(
                f"{packet_label} PR106 section {name} does not Brotli-decompress; "
                "refusing non-runtime-consumable monolithic candidate"
            ) from exc
        seen.add(name)
    missing = sorted(PR106_BROTLI_SECTION_NAMES - seen)
    if missing:
        raise MonolithicPacketCandidateError(
            f"{packet_label} PR106 layout missing Brotli section(s): {', '.join(missing)}"
        )


def _runtime_parity_summary(
    report: Mapping[str, Any] | None,
    *,
    candidate_archive_sha256: str,
    new_member_sha256: str,
    replacement_manifests: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "declared": isinstance(report, Mapping),
        "ready_flag": False,
        "artifact_bound": False,
        "matched_artifact_field": "",
    }
    blockers: list[str] = []
    if not isinstance(report, Mapping):
        return summary, ["runtime_consumption_proof_missing"]

    if report.get("schema") != RUNTIME_CONSUMPTION_PROOF_SCHEMA:
        blockers.append("runtime_consumption_proof_schema_missing_or_invalid")

    ready_field = next((field for field in RUNTIME_READY_FIELDS if report.get(field) is True), "")
    summary["ready_flag"] = bool(ready_field)
    summary["ready_field"] = ready_field
    if not ready_field:
        blockers.append("runtime_consumption_ready_flag_missing")

    archive_bound = _sha_matches(report.get("candidate_archive_sha256"), candidate_archive_sha256)
    matched_member_field = next(
        (
            field
            for field in RUNTIME_MEMBER_SHA_FIELDS
            if _sha_matches(report.get(field), new_member_sha256)
        ),
        "",
    )
    member_bound = bool(matched_member_field)
    summary["artifact_bound"] = archive_bound
    summary["member_bound"] = member_bound
    summary["matched_artifact_field"] = "candidate_archive_sha256" if archive_bound else ""
    summary["matched_member_field"] = matched_member_field
    summary["expected_candidate_archive_sha256"] = candidate_archive_sha256
    summary["expected_new_member_sha256"] = new_member_sha256
    if not archive_bound:
        blockers.append("runtime_consumption_candidate_archive_sha_mismatch_or_missing")
    if not member_bound:
        blockers.append("runtime_consumption_rebuilt_member_sha_mismatch_or_missing")

    consumed_sections, section_blockers = _runtime_changed_section_summary(
        report,
        replacement_manifests,
    )
    summary["changed_sections_bound"] = consumed_sections
    blockers.extend(section_blockers)

    for field in ("command_sha256", "log_sha256"):
        if not _is_sha256(report.get(field)):
            blockers.append(f"runtime_consumption_{field}_missing_or_invalid")

    return summary, blockers


def _runtime_changed_section_summary(
    report: Mapping[str, Any],
    replacement_manifests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_sections = report.get("changed_sections")
    section_map: dict[str, str] = {}
    if isinstance(raw_sections, Mapping):
        section_map = {
            str(name): str(value)
            for name, value in raw_sections.items()
            if isinstance(name, str) and isinstance(value, str)
        }
    elif isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, Mapping):
                continue
            name = item.get("section_name", item.get("name"))
            sha = item.get("new_sha256", item.get("sha256"))
            if isinstance(name, str) and isinstance(sha, str):
                section_map[name] = sha

    consumed: list[dict[str, Any]] = []
    blockers: list[str] = []
    for replacement in replacement_manifests:
        section_name = str(replacement["section_name"])
        expected_sha = str(replacement["new_sha256"])
        actual_sha = section_map.get(section_name, "")
        matched = actual_sha.lower() == expected_sha.lower()
        consumed.append(
            {
                "section_name": section_name,
                "expected_new_sha256": expected_sha,
                "reported_new_sha256": actual_sha,
                "matched": matched,
            }
        )
        if not matched:
            blockers.append(f"runtime_consumption_changed_section_missing_or_mismatch:{section_name}")
    return consumed, blockers


def _lane_claim_summary(
    report: Mapping[str, Any] | None,
    *,
    expected_lane_id: str | None,
    expected_instance_job_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "declared": isinstance(report, Mapping),
        "active": False,
        "lane_id": "",
        "instance_job_id": "",
        "claim_status": "",
        "claims_path": "",
    }
    blockers: list[str] = []
    if not isinstance(report, Mapping):
        return summary, ["active_lane_claim_missing"]

    if report.get("schema") != ACTIVE_LANE_CLAIM_JSON_SCHEMA:
        blockers.append("active_lane_claim_schema_missing_or_invalid")
    if report.get("blockers") not in (None, []):
        blockers.append("active_lane_claim_export_report_has_blockers")

    lane_id = report.get("lane_id")
    instance_job_id = report.get("instance_job_id", report.get("job_name"))
    claim_status = report.get("claim_status", report.get("status"))
    claims_path = report.get("claims_path")
    claimed_with = report.get("claimed_with", "")
    summary.update(
        {
            "active": report.get("active") is True,
            "lane_id": lane_id if isinstance(lane_id, str) else "",
            "instance_job_id": instance_job_id if isinstance(instance_job_id, str) else "",
            "claim_status": claim_status if isinstance(claim_status, str) else "",
            "claims_path": claims_path if isinstance(claims_path, str) else "",
        }
    )

    if not isinstance(lane_id, str) or not lane_id:
        blockers.append("active_lane_claim_lane_id_missing")
    if not isinstance(instance_job_id, str) or not instance_job_id:
        blockers.append("active_lane_claim_instance_job_id_missing")
    if not isinstance(claim_status, str) or not claim_status:
        blockers.append("active_lane_claim_status_missing")
    elif _status_is_terminal(claim_status):
        blockers.append("active_lane_claim_status_terminal")
    if report.get("active") is not True:
        blockers.append("active_lane_claim_not_active")
    if claims_path != CANONICAL_DISPATCH_CLAIMS_PATH:
        blockers.append("active_lane_claim_claims_path_not_canonical")
    if expected_lane_id is None:
        blockers.append("active_lane_claim_expected_lane_id_missing")
    elif lane_id != expected_lane_id:
        blockers.append("active_lane_claim_lane_id_mismatch")
    if expected_instance_job_id is None:
        blockers.append("active_lane_claim_expected_instance_job_id_missing")
    elif instance_job_id != expected_instance_job_id:
        blockers.append("active_lane_claim_instance_job_id_mismatch")
    if (
        not isinstance(claimed_with, str)
        or "tools/claim_lane_dispatch.py" not in claimed_with
        or " claim" not in f" {claimed_with} "
    ):
        blockers.append("active_lane_claim_helper_not_represented")
    row_hash = report.get("claim_row_sha256")
    if not _is_sha256(row_hash):
        blockers.append("active_lane_claim_row_sha256_missing_or_invalid")
    elif isinstance(claims_path, str) and not _claim_file_contains_row_hash(Path(claims_path), row_hash):
        blockers.append("active_lane_claim_row_not_found_in_claims_file")

    return summary, blockers


def _status_is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_CLAIM_PREFIXES)


def _sha_matches(value: Any, expected: str) -> bool:
    return isinstance(value, str) and value.lower() == expected.lower()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _claim_file_contains_row_hash(path: Path, row_hash: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(hashlib.sha256(line.encode("utf-8")).hexdigest() == row_hash for line in text.splitlines())


def _rewrite_pr106_header(member: bytes, sections: list[dict[str, Any]]) -> bytes:
    decoder_section = next(
        (section for section in sections if section["name"] == "decoder_packed_brotli"),
        None,
    )
    if decoder_section is None:
        raise MonolithicPacketCandidateError("PR106 layout missing decoder_packed_brotli")
    decoder_len = int(decoder_section["new_len"])
    if decoder_len <= 0 or decoder_len >= 2**24:
        raise MonolithicPacketCandidateError(f"invalid PR106 decoder section length: {decoder_len}")
    return bytes([0xFF]) + decoder_len.to_bytes(3, "little") + member[4:]


def _refresh_pr106_header_section(
    member: bytes,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    cursor = 0
    for section in sections:
        new_len = int(section["new_len"])
        if section["name"] == "ff_header":
            new_len = 4
            section = {
                **section,
                "new_len": 4,
                "new_sha256": sha256_bytes(member[:4]),
                "changed": section["old_sha256"] != sha256_bytes(member[:4]),
            }
        refreshed.append({**section, "new_offset": cursor})
        cursor += new_len
    return refreshed


def _write_single_member_zip(path: Path, *, member_name: str, payload: bytes) -> None:
    _validate_zip_member_name(member_name)
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.flag_bits = 0
    info.internal_attr = 0
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _validate_zip_member_name(member_name: str) -> None:
    if not member_name:
        raise MonolithicPacketCandidateError("ZIP member name is empty")
    if member_name.startswith(("/", "\\")) or "\\" in member_name:
        raise MonolithicPacketCandidateError(f"unsafe ZIP member name: {member_name!r}")
    parts = member_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise MonolithicPacketCandidateError(f"unsafe ZIP member name: {member_name!r}")
    if any(part.startswith(".") for part in parts):
        raise MonolithicPacketCandidateError(f"hidden ZIP member name rejected: {member_name!r}")
