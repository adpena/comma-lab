#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prune receiver-inert metadata from a SNeRV SNAR1 packet."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="Raw SNAR1 packet or archive.zip containing member 0.bin.",
    )
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--output-archive-zip",
        type=Path,
        help="Optional deterministic contest archive.zip with stored 0.bin.",
    )
    parser.add_argument("--hard-byte-ceiling", action="append", type=int, default=[])
    parser.add_argument("--pair-index", action="append", type=int, default=[])
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
    source_packet, source_kind = _read_source_packet(packet_path)
    report, candidate_packet = build_snerv_snar_header_minimization(
        source_packet,
        source_packet_path=_source_path_label(packet_path, source_kind),
        proof_pair_indices=tuple(int(value) for value in args.pair_index),
        frame_proof_max_output_bytes=int(args.frame_proof_max_output_bytes),
        hard_byte_ceilings=tuple(int(value) for value in args.hard_byte_ceiling),
        raw_argv=raw_argv,
    )
    report["source_packet"]["input_kind"] = source_kind
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
        archive_bytes = _build_deterministic_archive_zip(candidate_packet)
        archive_result = write_bytes_artifact(args.output_archive_zip, archive_bytes)
        archive_result_payload = {
            "path": archive_result.path,
            "bytes": archive_result.bytes_written,
            "sha256": archive_result.sha256,
            "member": "0.bin",
            "member_bytes": len(candidate_packet),
            "member_sha256": report["candidate_packet"]["sha256"],
            "zip_compression": "stored",
            "deterministic_mtime": "1980-01-01T00:00:00",
        }
        report["candidate_archive_zip"] = archive_result_payload
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
    json_result = write_json_artifact(args.output_json, report)
    report["report_path"] = json_result.path
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_packet": packet_result.path,
                "output_json": json_result.path,
                "source_packet_bytes": report["source_packet"]["bytes"],
                "candidate_packet_bytes": report["candidate_packet"]["bytes"],
                "candidate_archive_zip_bytes": (
                    None
                    if archive_result_payload is None
                    else archive_result_payload["bytes"]
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
                "receiver_contract_satisfied": report[
                    "receiver_contract_satisfied"
                ],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_source_packet(path: Path) -> tuple[bytes, str]:
    blob = path.read_bytes()
    if blob.startswith(b"SNAR1"):
        return blob, "raw_snar_packet"
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "0.bin" not in names:
                raise ValueError(f"{path}: archive.zip missing required 0.bin member")
            return archive.read("0.bin"), "archive_zip_member_0_bin"
    raise ValueError(f"{path}: expected raw SNAR1 packet or archive.zip")


def _source_path_label(path: Path, source_kind: str) -> str:
    if source_kind == "archive_zip_member_0_bin":
        return f"{path.as_posix()}::0.bin"
    return path.as_posix()


def _build_deterministic_archive_zip(packet: bytes) -> bytes:
    out = io.BytesIO()
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(info, bytes(packet))
    return out.getvalue()


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


if __name__ == "__main__":
    raise SystemExit(main())
