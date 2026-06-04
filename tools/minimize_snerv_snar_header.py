#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prune receiver-inert metadata from a SNeRV receiver packet."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_snar_header_minimizer import (  # noqa: E402
    DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    SCHEMA,
    build_snerv_snar_header_minimization,
)
from tac.repo_io import write_bytes_artifact, write_json_artifact  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    SNERV_RECEIVER_PROOF_SCHEMA,
    export_snerv_archive_bound_candidate_package,
)


@dataclass(frozen=True)
class _ZipMember:
    info: zipfile.ZipInfo
    data: bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="Raw SNAR1/SNAR2 packet or archive.zip containing member 0.bin.",
    )
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--wire-format",
        choices=("snar1", "snar2"),
        default="snar1",
        help=(
            "Output packet grammar. snar1 preserves the JSON outer header; "
            "snar2 emits the fixed binary receiver header."
        ),
    )
    parser.add_argument(
        "--candidate-id",
        help=(
            "Optional planner/campaign candidate id. When omitted the report "
            "stays packet-SHA-only and cannot re-enable launch rows."
        ),
    )
    parser.add_argument(
        "--output-archive-zip",
        type=Path,
        help=(
            "Optional deterministic archive.zip. Raw SNAR input emits a packet-only "
            "0.bin zip; archive.zip input preserves all non-0.bin runtime members "
            "content-exact and replaces only 0.bin."
        ),
    )
    parser.add_argument("--hard-byte-ceiling", action="append", type=int, default=[])
    parser.add_argument("--pair-index", action="append", type=int, default=[])
    parser.add_argument(
        "--full-video-receiver-proof",
        action="store_true",
        help=(
            "Stream all pairs through source and minimized packets to prove "
            "receiver frame equality without materializing the full video tensor."
        ),
    )
    parser.add_argument(
        "--output-package-dir",
        type=Path,
        help=(
            "Optional SSD-backed contest-shaped runtime package directory. Runs "
            "generated inflate.sh against the minimized packet and writes a "
            "receiver proof while keeping score/eval authority false."
        ),
    )
    parser.add_argument(
        "--retain-package-raw",
        action="store_true",
        help="Keep the generated package proof raw output instead of certify-and-delete.",
    )
    parser.add_argument(
        "--package-timeout-seconds",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--source-receiver-proof-json",
        type=Path,
        help=(
            "Optional receiver_proof JSON from the source archive. When provided "
            "with --output-package-dir, the minimized package output SHA/bytes "
            "must match it to clear full-video signal-loss blockers."
        ),
    )
    parser.add_argument(
        "--frame-proof-max-output-bytes",
        type=int,
        default=DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    packet_path = args.packet.expanduser().resolve(strict=False)
    source_input_blob = packet_path.read_bytes()
    source_packet, source_kind, source_archive_members = _read_source_packet(packet_path)
    report, candidate_packet = build_snerv_snar_header_minimization(
        source_packet,
        source_packet_path=_source_path_label(packet_path, source_kind),
        candidate_id=args.candidate_id,
        wire_format=args.wire_format,
        proof_pair_indices=tuple(int(value) for value in args.pair_index),
        full_video_receiver_proof=bool(args.full_video_receiver_proof),
        frame_proof_max_output_bytes=int(args.frame_proof_max_output_bytes),
        hard_byte_ceilings=tuple(int(value) for value in args.hard_byte_ceiling),
        raw_argv=raw_argv,
    )
    report["source_packet"]["input_kind"] = source_kind
    report["source_input"] = {
        "path": packet_path.as_posix(),
        "kind": source_kind,
        "bytes": len(source_input_blob),
        "sha256": _sha256(source_input_blob),
    }
    packet_result = write_bytes_artifact(args.output_packet, candidate_packet)
    report["candidate_packet"]["path"] = packet_result.path
    report["candidate_packet"]["file_bytes"] = packet_result.bytes_written
    report["candidate_packet"]["file_sha256"] = packet_result.sha256
    report["candidate_packet"]["file_matches_report"] = (
        report["candidate_packet"]["bytes"] == packet_result.bytes_written
        and report["candidate_packet"]["sha256"] == packet_result.sha256
    )
    archive_result_payload: dict[str, object] | None = None
    if args.output_archive_zip is not None:
        archive_bytes, archive_manifest = _build_deterministic_archive_zip(
            candidate_packet,
            source_members=source_archive_members,
        )
        archive_result = write_bytes_artifact(args.output_archive_zip, archive_bytes)
        archive_result_payload = {
            **archive_manifest,
            "path": archive_result.path,
            "bytes": archive_result.bytes_written,
            "sha256": archive_result.sha256,
            "member_bytes": len(candidate_packet),
            "member_sha256": report["candidate_packet"]["sha256"],
        }
        report["candidate_archive_zip"] = archive_result_payload
        report["candidate_archive_zip_runtime_preserving_repack"] = archive_result_payload
        report["contest_compliance_contract"]["archive_zip_materialized"] = True
        report["contest_compliance_contract"]["archive_zip_bytes"] = (
            archive_result.bytes_written
        )
        report["blockers"] = [
            blocker
            for blocker in report["blockers"]
            if blocker != "not_packaged_as_contest_archive_zip"
        ]
        _attach_archive_byte_ceiling_rows(report, archive_result.bytes_written)
    if args.output_package_dir is not None:
        package = export_snerv_archive_bound_candidate_package(
            packet=candidate_packet,
            output_dir=args.output_package_dir,
            repo_root=REPO_ROOT,
            retain_receiver_output=bool(args.retain_package_raw),
            receiver_proof_timeout_seconds=int(args.package_timeout_seconds),
            mlx_triage_argv=raw_argv,
        )
        _attach_runtime_package_report(report, package)
        if args.source_receiver_proof_json is not None:
            source_proof = json.loads(
                args.source_receiver_proof_json.expanduser()
                .resolve(strict=False)
                .read_text(encoding="utf-8")
            )
            _attach_source_receiver_output_parity(
                report,
                source_proof=source_proof,
                source_proof_path=args.source_receiver_proof_json,
            )
    json_result = write_json_artifact(args.output_json, report)
    report["report_path"] = json_result.path
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_packet": packet_result.path,
                "output_json": json_result.path,
                "source_packet_bytes": report["source_packet"]["bytes"],
                "candidate_id": report["candidate_binding"]["candidate_id"],
                "wire_format": report["wire_format"],
                "candidate_packet_bytes": report["candidate_packet"]["bytes"],
                "candidate_archive_zip_bytes": (
                    None
                    if report.get("candidate_archive_zip") is None
                    else report["candidate_archive_zip"].get("bytes")
                ),
                "candidate_archive_zip_kind": (
                    None
                    if report.get("candidate_archive_zip") is None
                    else report["candidate_archive_zip"].get("archive_zip_kind")
                ),
                "runtime_consumption_proof_passed": (
                    report.get("runtime_package", {})
                    .get("receiver_proof", {})
                    .get("runtime_consumption_proof_passed")
                ),
                "source_receiver_output_parity_status": (
                    report.get("source_receiver_output_parity", {}).get("status")
                ),
                "packet_byte_delta": report["packet_byte_delta"],
                "source_header_bytes": report["source_packet"]["header_bytes"],
                "candidate_header_bytes": report["candidate_packet"]["header_bytes"],
                "header_byte_delta": report["header_byte_delta"],
                "removed_metadata_json_bytes": report["removed_metadata"][
                    "json_bytes"
                ],
                "receiver_pair_frame_equality_status": report[
                    "receiver_pair_frame_equality_proof"
                ]["status"],
                "receiver_pair_frame_equality_scope": report[
                    "receiver_pair_frame_equality_proof"
                ].get("scope"),
                "receiver_contract_satisfied": report[
                    "receiver_contract_satisfied"
                ],
                "full_video_receiver_contract_satisfied": report[
                    "full_video_receiver_contract_satisfied"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_source_packet(path: Path) -> tuple[bytes, str, tuple[_ZipMember, ...]]:
    blob = path.read_bytes()
    if blob.startswith(b"SNAR1"):
        return blob, "raw_snar1_packet", ()
    if blob.startswith(b"SNAR2"):
        return blob, "raw_snar2_packet", ()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(set(names)) != len(names):
                raise ValueError(
                    f"{path}: archive.zip has duplicate members; refusing ambiguous "
                    "runtime-preserving rewrite"
                )
            if names.count("0.bin") != 1:
                raise ValueError(f"{path}: archive.zip missing required 0.bin member")
            members = tuple(
                _ZipMember(info=info, data=archive.open(info).read())
                for info in infos
            )
            packet = next(member.data for member in members if member.info.filename == "0.bin")
            return packet, "archive_zip_member_0_bin", members
    raise ValueError(f"{path}: expected raw SNAR1/SNAR2 packet or archive.zip")


def _source_path_label(path: Path, source_kind: str) -> str:
    if source_kind == "archive_zip_member_0_bin":
        return f"{path.as_posix()}::0.bin"
    return path.as_posix()


def _build_deterministic_archive_zip(
    packet: bytes,
    *,
    source_members: tuple[_ZipMember, ...] = (),
) -> tuple[bytes, dict[str, object]]:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        if source_members:
            rows: list[dict[str, object]] = []
            for member in source_members:
                name = member.info.filename
                payload = bytes(packet) if name == "0.bin" else member.data
                info = _clone_zip_info(member.info)
                archive.writestr(info, payload)
                rows.append(_zip_member_row(info, payload, replaced=name == "0.bin"))
            archive_kind = "runtime_preserving_repack"
        else:
            info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, bytes(packet))
            rows = [_zip_member_row(info, bytes(packet), replaced=True)]
            archive_kind = "packet_only_zip"
    payload = out.getvalue()
    return payload, {
        "archive_zip_kind": archive_kind,
        "member": "0.bin",
        "member_count": len(rows),
        "member_rows": rows,
        "runtime_preserved_from_source_archive": bool(source_members),
        "non_0bin_members_content_exact_from_source": bool(source_members),
    }


def _clone_zip_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=source.filename, date_time=source.date_time)
    info.compress_type = source.compress_type
    info.comment = source.comment
    info.extra = source.extra
    info.internal_attr = source.internal_attr
    info.external_attr = source.external_attr
    info.create_system = source.create_system
    return info


def _zip_member_row(
    info: zipfile.ZipInfo,
    payload: bytes,
    *,
    replaced: bool,
) -> dict[str, object]:
    return {
        "filename": info.filename,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "compress_type": int(info.compress_type),
        "compression": (
            "stored" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type)
        ),
        "mtime": (
            f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d}"
            f"T{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}"
        ),
        "replaced_by_minimized_packet": bool(replaced),
    }


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


def _attach_archive_byte_ceiling_rows(
    report: dict[str, object],
    archive_zip_bytes: int,
) -> None:
    rows = report.get("hard_byte_ceiling_rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        ceiling = int(row.get("hard_byte_ceiling") or 0)
        row["candidate_archive_zip_bytes"] = int(archive_zip_bytes)
        row["candidate_archive_zip_over_ceiling_bytes"] = max(
            int(archive_zip_bytes) - ceiling,
            0,
        )
        row["candidate_archive_zip_under_ceiling"] = (
            int(archive_zip_bytes) <= ceiling
        )


def _attach_runtime_package_report(
    report: dict[str, object],
    package: dict[str, object],
) -> None:
    proof = dict(package.get("receiver_proof") or {})
    adapter_package = dict(package.get("archive_bound_candidate_adapter_package") or {})
    rows = list(adapter_package.get("candidate_rows") or ())
    first_row = dict(rows[0]) if rows and isinstance(rows[0], dict) else {}
    archive_path = first_row.get("candidate_archive_path") or proof.get("archive_path")
    archive_bytes = _positive_int(
        first_row.get("candidate_archive_bytes") or proof.get("archive_bytes")
    )
    archive_sha256 = first_row.get("candidate_archive_sha256") or proof.get(
        "archive_sha256"
    )
    report["runtime_package"] = {
        "schema": package.get("schema"),
        "package_path": _runtime_package_path(proof),
        "candidate_row_schema": first_row.get("schema"),
        "candidate_id": first_row.get("candidate_id"),
        "candidate_archive_path": archive_path,
        "candidate_archive_bytes": archive_bytes,
        "candidate_archive_sha256": archive_sha256,
        "runtime_consumption_proof_path": first_row.get(
            "runtime_consumption_proof_path"
        )
        or proof.get("proof_path"),
        "receiver_proof": {
            "schema": proof.get("schema"),
            "proof_path": proof.get("proof_path"),
            "archive_path": proof.get("archive_path"),
            "archive_bytes": proof.get("archive_bytes"),
            "archive_sha256": proof.get("archive_sha256"),
            "runtime_consumption_proof_passed": (
                proof.get("runtime_consumption_proof_passed") is True
            ),
            "receiver_contract_satisfied": (
                proof.get("receiver_contract_satisfied") is True
            ),
            "receiver_output_bytes": proof.get("receiver_output_bytes"),
            "receiver_output_sha256": proof.get("receiver_output_sha256"),
            "receiver_output_retained": proof.get("receiver_output_retained"),
            "blockers": list(proof.get("blockers") or ()),
        },
        "blockers": list(first_row.get("blockers") or ()),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    report["contest_compliance_contract"]["runtime_package_materialized"] = True
    report["contest_compliance_contract"]["runtime_consumption_proof_passed"] = (
        proof.get("runtime_consumption_proof_passed") is True
    )
    if proof.get("runtime_consumption_proof_passed") is True:
        report["runtime_consumption_proof_ready"] = True
        report["blockers"] = [
            blocker
            for blocker in report["blockers"]
            if blocker != "snerv_snar_header_minimization_receiver_proof_failed"
        ]
    if archive_path and archive_bytes is not None and archive_sha256:
        existing_archive = report.get("candidate_archive_zip")
        if isinstance(existing_archive, dict) and existing_archive.get("path") != archive_path:
            report["candidate_archive_zip_pre_package_repack"] = dict(existing_archive)
        report["candidate_archive_zip"] = {
            "path": archive_path,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "member": "0.bin",
            "archive_zip_kind": "generated_runtime_package",
            "runtime_preserved_from_source_archive": False,
            "runtime_generated_by_package_exporter": True,
        }
        report["contest_compliance_contract"]["archive_zip_materialized"] = True
        report["contest_compliance_contract"]["archive_zip_bytes"] = archive_bytes
        report["blockers"] = [
            blocker
            for blocker in report["blockers"]
            if blocker != "not_packaged_as_contest_archive_zip"
        ]
        _attach_archive_byte_ceiling_rows(report, archive_bytes)


def _attach_source_receiver_output_parity(
    report: dict[str, object],
    *,
    source_proof: dict[str, object],
    source_proof_path: Path,
) -> None:
    runtime_package = report.get("runtime_package")
    candidate_proof = (
        runtime_package.get("receiver_proof")
        if isinstance(runtime_package, dict)
        else None
    )
    if not isinstance(candidate_proof, dict):
        report["source_receiver_output_parity"] = {
            "status": "missing_candidate_runtime_proof",
            "source_proof_path": source_proof_path.as_posix(),
            "exact_equal": False,
            "blockers": ["snerv_snar_header_minimization_candidate_runtime_proof_missing"],
        }
        _append_unique_blocker(
            report,
            "snerv_snar_header_minimization_source_receiver_output_parity_missing",
        )
        return

    source_proof_file_sha256 = _path_sha256(source_proof_path)
    source_bytes = _positive_int(source_proof.get("receiver_output_bytes"))
    candidate_bytes = _positive_int(candidate_proof.get("receiver_output_bytes"))
    source_sha = str(source_proof.get("receiver_output_sha256") or "").strip()
    candidate_sha = str(candidate_proof.get("receiver_output_sha256") or "").strip()
    source_schema_ok = source_proof.get("schema") == SNERV_RECEIVER_PROOF_SCHEMA
    candidate_schema_ok = candidate_proof.get("schema") == SNERV_RECEIVER_PROOF_SCHEMA
    source_ready = source_proof.get("runtime_consumption_proof_passed") is True
    candidate_ready = candidate_proof.get("runtime_consumption_proof_passed") is True
    source_identity = _source_receiver_proof_identity(report, source_proof)
    candidate_identity = _candidate_receiver_proof_identity(report, candidate_proof)
    exact = bool(
        source_schema_ok
        and candidate_schema_ok
        and source_identity["exact_equal"]
        and candidate_identity["exact_equal"]
        and source_ready
        and candidate_ready
        and source_bytes is not None
        and source_bytes == candidate_bytes
        and source_sha
        and source_sha == candidate_sha
    )
    report["source_receiver_output_parity"] = {
        "status": "proven_exact" if exact else "failed",
        "source_proof_path": source_proof_path.as_posix(),
        "source_proof_file_sha256": source_proof_file_sha256,
        "source_proof_schema": source_proof.get("schema"),
        "candidate_proof_path": candidate_proof.get("proof_path"),
        "candidate_proof_schema": candidate_proof.get("schema"),
        "source_receiver_output_bytes": source_bytes,
        "candidate_receiver_output_bytes": candidate_bytes,
        "source_receiver_output_sha256": source_sha or None,
        "candidate_receiver_output_sha256": candidate_sha or None,
        "source_runtime_consumption_proof_passed": bool(source_ready),
        "candidate_runtime_consumption_proof_passed": bool(candidate_ready),
        "source_proof_schema_valid": bool(source_schema_ok),
        "candidate_proof_schema_valid": bool(candidate_schema_ok),
        "source_archive_identity": source_identity,
        "candidate_archive_identity": candidate_identity,
        "exact_equal": exact,
        "blockers": []
        if exact
        else _source_receiver_parity_blockers(
            source_schema_ok=source_schema_ok,
            candidate_schema_ok=candidate_schema_ok,
            source_ready=source_ready,
            candidate_ready=candidate_ready,
            source_identity=source_identity,
            candidate_identity=candidate_identity,
            output_equal=bool(
                source_bytes is not None
                and source_bytes == candidate_bytes
                and source_sha
                and source_sha == candidate_sha
            ),
        ),
    }
    if exact:
        report["full_video_receiver_contract_satisfied"] = True
        report["runtime_consumption_proof_ready"] = True
        report["blockers"] = [
            blocker
            for blocker in report["blockers"]
            if blocker
            not in {
                "snerv_snar_header_minimization_receiver_proof_failed",
                "snerv_snar_header_minimization_full_video_receiver_proof_missing",
            }
        ]
        report["contest_compliance_contract"][
            "source_receiver_output_parity_proven"
        ] = True
    else:
        _append_unique_blocker(
            report,
            "snerv_snar_header_minimization_source_receiver_output_parity_failed",
        )
        for blocker in report["source_receiver_output_parity"]["blockers"]:
            _append_unique_blocker(report, str(blocker))


def _append_unique_blocker(report: dict[str, object], blocker: str) -> None:
    blockers = list(report.get("blockers") or ())
    if blocker not in blockers:
        blockers.append(blocker)
    report["blockers"] = blockers


def _source_receiver_proof_identity(
    report: dict[str, object],
    source_proof: dict[str, object],
) -> dict[str, object]:
    proof_archive_sha = str(source_proof.get("archive_sha256") or "").strip()
    source_input = report.get("source_input")
    source_input_sha = (
        str(source_input.get("sha256") or "").strip()
        if isinstance(source_input, dict)
        else ""
    )
    source_kind = (
        str(source_input.get("kind") or "").strip() if isinstance(source_input, dict) else ""
    )
    identity_kind = (
        "source_archive_zip_sha256"
        if source_kind == "archive_zip_member_0_bin"
        else "source_raw_packet_has_no_archive_proof_identity"
    )
    exact = bool(
        source_kind == "archive_zip_member_0_bin"
        and proof_archive_sha
        and proof_archive_sha == source_input_sha
    )
    return {
        "identity_kind": identity_kind,
        "expected_archive_sha256": source_input_sha or None,
        "proof_archive_sha256": proof_archive_sha or None,
        "exact_equal": exact,
    }


def _candidate_receiver_proof_identity(
    report: dict[str, object],
    candidate_proof: dict[str, object],
) -> dict[str, object]:
    candidate_archive = report.get("candidate_archive_zip")
    expected_sha = (
        str(candidate_archive.get("sha256") or "").strip()
        if isinstance(candidate_archive, dict)
        else ""
    )
    proof_archive_sha = str(candidate_proof.get("archive_sha256") or "").strip()
    exact = bool(expected_sha and proof_archive_sha and expected_sha == proof_archive_sha)
    return {
        "identity_kind": "candidate_archive_zip_sha256",
        "expected_archive_sha256": expected_sha or None,
        "proof_archive_sha256": proof_archive_sha or None,
        "exact_equal": exact,
    }


def _source_receiver_parity_blockers(
    *,
    source_schema_ok: bool,
    candidate_schema_ok: bool,
    source_ready: bool,
    candidate_ready: bool,
    source_identity: dict[str, object],
    candidate_identity: dict[str, object],
    output_equal: bool,
) -> list[str]:
    blockers: list[str] = []
    if not source_schema_ok:
        blockers.append("snerv_source_receiver_proof_schema_invalid")
    if not candidate_schema_ok:
        blockers.append("snerv_candidate_receiver_proof_schema_invalid")
    if not source_ready:
        blockers.append("snerv_source_receiver_runtime_proof_not_passed")
    if not candidate_ready:
        blockers.append("snerv_candidate_receiver_runtime_proof_not_passed")
    if source_identity.get("exact_equal") is not True:
        blockers.append("snerv_source_receiver_proof_archive_identity_mismatch")
    if candidate_identity.get("exact_equal") is not True:
        blockers.append("snerv_candidate_receiver_proof_archive_identity_mismatch")
    if not output_equal:
        blockers.append("snerv_snar_header_minimization_source_receiver_output_changed")
    return blockers or ["snerv_snar_header_minimization_source_receiver_output_changed"]


def _runtime_package_path(proof: dict[str, object]) -> str | None:
    proof_path = str(proof.get("proof_path") or "").strip()
    if not proof_path:
        return None
    path = Path(proof_path)
    if path.name:
        return path.parents[1].as_posix() if len(path.parents) > 1 else path.as_posix()
    return None


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _path_sha256(path: Path) -> str | None:
    try:
        return _sha256(path.expanduser().resolve(strict=False).read_bytes())
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
