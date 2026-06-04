#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an exact SNeRV step-map packet compaction candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_step_map_coder import (  # noqa: E402
    decode_step_maps,
    encode_step_maps_adaptive,
)
from tac.repo_io import sha256_file  # noqa: E402
from tac.substrates._shared.pact_nerv_full_main import build_archive_zip  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    SNERV_ARCHIVE_SCHEMA_V2,
    pack_snerv_archive,
    pack_snerv_archive_snar2,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    export_snerv_archive_bound_candidate_package,
)

SCHEMA = "snerv_snar_step_map_compaction_materialization.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_path = args.packet.expanduser().resolve(strict=False)
    source_packet, source_kind, zip_members = _read_packet(source_path)
    report, candidate_packet = build_snerv_step_map_compaction(
        source_packet,
        source_path=source_path.as_posix(),
        source_kind=source_kind,
        source_zip_members=zip_members,
        candidate_id=args.candidate_id,
        wire_format=args.wire_format,
        hard_byte_ceilings=tuple(args.hard_byte_ceiling or ()),
        generated_utc=datetime.now(UTC).isoformat(),
    )
    if not report["decoded_step_maps_exact_equal"]:
        _write_report(args.output_json, report)
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "output_json": Path(args.output_json).as_posix(),
                    "decoded_step_maps_exact_equal": False,
                    "blockers": report["blockers"],
                    **FALSE_AUTHORITY,
                },
                sort_keys=True,
            )
        )
        return 2

    output_packet = args.output_packet.expanduser().resolve(strict=False)
    output_packet.parent.mkdir(parents=True, exist_ok=True)
    output_packet.write_bytes(candidate_packet)
    report["candidate_packet"]["path"] = output_packet.as_posix()

    archive_zip = None
    if args.output_archive_zip is not None and args.output_package_dir is None:
        archive_zip = args.output_archive_zip.expanduser().resolve(strict=False)
        archive_zip.parent.mkdir(parents=True, exist_ok=True)
        build_archive_zip(archive_zip, bin_bytes=candidate_packet)
        report["candidate_archive_zip"] = {
            "path": archive_zip.as_posix(),
            "bytes": archive_zip.stat().st_size,
            "sha256": sha256_file(archive_zip),
            "member": "0.bin",
            "archive_zip_kind": "packet_only_no_generated_runtime",
        }

    if args.full_video_receiver_proof:
        if args.output_package_dir is None:
            _write_blocked_report(
                args.output_json,
                report,
                "snerv_step_map_compaction_output_package_dir_missing",
            )
            return 2
        package = export_snerv_archive_bound_candidate_package(
            packet=candidate_packet,
            output_dir=args.output_package_dir,
            retain_receiver_output=bool(args.retain_receiver_output),
            receiver_proof_timeout_seconds=int(args.package_timeout_seconds),
        )
        _attach_runtime_package_report(report, package)
        if args.output_archive_zip is not None:
            archive_zip = args.output_archive_zip.expanduser().resolve(strict=False)
            archive_zip.parent.mkdir(parents=True, exist_ok=True)
            package_archive = Path(report["receiver_package"]["archive_zip_path"])
            shutil.copyfile(package_archive, archive_zip)
            report["candidate_archive_zip"] = {
                "path": archive_zip.as_posix(),
                "bytes": archive_zip.stat().st_size,
                "sha256": sha256_file(archive_zip),
                "member": "0.bin",
                "archive_zip_kind": "generated_runtime_package_copy",
            }

    if not report.get("runtime_consumption_proof_passed"):
        report["blockers"] = _dedupe(
            [
                *list(report.get("blockers") or ()),
                "snerv_step_map_compaction_receiver_proof_missing",
            ]
        )

    _write_report(args.output_json, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": Path(args.output_json).as_posix(),
                "output_packet": output_packet.as_posix(),
                "source_packet_bytes": report["source_packet"]["bytes"],
                "candidate_packet_bytes": report["candidate_packet"]["bytes"],
                "packet_byte_delta": report["packet_byte_delta"],
                "step_map_packet_byte_delta": report["section_bytes"][
                    "step_map_packet_delta"
                ],
                "decoded_step_maps_exact_equal": report[
                    "decoded_step_maps_exact_equal"
                ],
                "runtime_consumption_proof_passed": report.get(
                    "runtime_consumption_proof_passed", False
                ),
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0


def build_snerv_step_map_compaction(
    packet: bytes,
    *,
    source_path: str | None,
    source_kind: str,
    source_zip_members: Sequence[Mapping[str, Any]] = (),
    candidate_id: str | None = None,
    wire_format: str = "snar2",
    hard_byte_ceilings: Sequence[int] = (),
    generated_utc: str,
) -> tuple[dict[str, Any], bytes]:
    source = unpack_snerv_archive(packet)
    source_step_packet = source.sections["step_map_packet"]
    source_maps = decode_step_maps(source_step_packet)
    recoded = encode_step_maps_adaptive(
        source_maps,
        map_importance=np.arange(len(source_maps), dtype=np.float64),
        bin_choices=(16, 4),
        constant_importance_quantile=1.0,
        binary_header=True,
    )
    recoded_maps = decode_step_maps(recoded.packet)
    exact = len(source_maps) == len(recoded_maps) and all(
        np.array_equal(left, right)
        for left, right in zip(source_maps, recoded_maps, strict=True)
    )
    if not exact:
        report = _base_report(
            source_packet=packet,
            source_path=source_path,
            source_kind=source_kind,
            source_zip_members=source_zip_members,
            source_schema=source.schema,
            source_step_packet=source_step_packet,
            recoded_step_packet=recoded.packet,
            candidate_packet=b"",
            candidate_schema=None,
            candidate_id=candidate_id,
            wire_format=wire_format,
            hard_byte_ceilings=hard_byte_ceilings,
            generated_utc=generated_utc,
            recoded_groups=recoded.groups,
            decoded_exact=False,
        )
        report["blockers"] = _dedupe(
            [
                *report["blockers"],
                "snerv_step_map_compaction_not_exact_for_source_step_maps",
            ]
        )
        return report, b""

    candidate_format = _output_wire_format(wire_format, source_schema=source.schema)
    sections = dict(source.sections)
    sections["step_map_packet"] = recoded.packet
    if candidate_format == "snar2":
        candidate = pack_snerv_archive_snar2(
            metadata_payload=sections["metadata_payload"],
            lf_payload=sections["lf_payload"],
            decoder_payload=sections["decoder_payload"],
            step_map_packet=sections["step_map_packet"],
            metadata=source.metadata,
        )
    else:
        candidate = pack_snerv_archive(
            metadata_payload=sections["metadata_payload"],
            lf_payload=sections["lf_payload"],
            decoder_payload=sections["decoder_payload"],
            step_map_packet=sections["step_map_packet"],
            metadata=source.metadata,
        )
    report = _base_report(
        source_packet=packet,
        source_path=source_path,
        source_kind=source_kind,
        source_zip_members=source_zip_members,
        source_schema=source.schema,
        source_step_packet=source_step_packet,
        recoded_step_packet=recoded.packet,
        candidate_packet=candidate.packet,
        candidate_schema=candidate.schema,
        candidate_id=candidate_id,
        wire_format=candidate_format,
        hard_byte_ceilings=hard_byte_ceilings,
        generated_utc=generated_utc,
        recoded_groups=recoded.groups,
        decoded_exact=True,
    )
    return report, candidate.packet


def _base_report(
    *,
    source_packet: bytes,
    source_path: str | None,
    source_kind: str,
    source_zip_members: Sequence[Mapping[str, Any]],
    source_schema: str,
    source_step_packet: bytes,
    recoded_step_packet: bytes,
    candidate_packet: bytes,
    candidate_schema: str | None,
    candidate_id: str | None,
    wire_format: str,
    hard_byte_ceilings: Sequence[int],
    generated_utc: str,
    recoded_groups: Sequence[Mapping[str, Any]],
    decoded_exact: bool,
) -> dict[str, Any]:
    candidate_packet_bytes = len(candidate_packet) if candidate_packet else None
    return {
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "axis_tag": "[receiver-safe:false-authority]",
        "operation": "snerv_step_map_constant_shape_partition_binary_repack",
        "wire_format": wire_format,
        "candidate_binding": {
            "candidate_id": candidate_id,
            "candidate_id_required_for_launch_reenable": True,
        },
        "source_packet": {
            "path": source_path,
            "kind": source_kind,
            "schema": source_schema,
            "bytes": len(source_packet),
            "sha256": _sha256(source_packet),
            "zip_members": list(source_zip_members),
        },
        "candidate_packet": {
            "path": None,
            "schema": candidate_schema,
            "bytes": candidate_packet_bytes,
            "sha256": None if not candidate_packet else _sha256(candidate_packet),
        },
        "packet_byte_delta": (
            None
            if candidate_packet_bytes is None
            else candidate_packet_bytes - len(source_packet)
        ),
        "section_bytes": {
            "source_step_map_packet": len(source_step_packet),
            "candidate_step_map_packet": len(recoded_step_packet),
            "step_map_packet_delta": len(recoded_step_packet)
            - len(source_step_packet),
        },
        "section_sha256": {
            "source_step_map_packet": _sha256(source_step_packet),
            "candidate_step_map_packet": _sha256(recoded_step_packet),
        },
        "recoded_group_summary": [
            {
                "kind": group.get("kind"),
                "map_count": len(group.get("map_indices") or ()),
                "shape": group.get("shape"),
                "payload_bytes": group.get("payload_bytes"),
                "raw_bytes": group.get("raw_bytes"),
                "legacy_shape_count": len(group.get("shapes") or ()),
            }
            for group in recoded_groups
        ],
        "decoded_step_maps_exact_equal": bool(decoded_exact),
        "runtime_consumption_proof_passed": False,
        "receiver_contract_satisfied": False,
        "hard_byte_ceiling_rows": _hard_byte_ceiling_rows(
            source_packet_bytes=len(source_packet),
            candidate_packet_bytes=candidate_packet_bytes,
            ceilings=hard_byte_ceilings,
        ),
        "blockers": [
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ],
        "next_actions": [
            "feed_materialized_step_map_compaction_report_back_to_over_ceiling_reroute_queue",
            "run_paired_contest_cpu_cuda_auth_eval_only_after_score_axis_packet_selection",
        ],
        **FALSE_AUTHORITY,
    }


def _attach_runtime_package_report(
    report: dict[str, Any],
    package: Mapping[str, Any],
) -> None:
    proof = dict(package.get("receiver_proof") or {})
    adapter_package = dict(package.get("archive_bound_candidate_adapter_package") or {})
    rows = list(adapter_package.get("candidate_rows") or ())
    first_row = dict(rows[0]) if rows and isinstance(rows[0], Mapping) else {}
    archive_path = first_row.get("candidate_archive_path") or proof.get("archive_path")
    archive_bytes = first_row.get("candidate_archive_bytes") or proof.get("archive_bytes")
    archive_sha256 = first_row.get("candidate_archive_sha256") or proof.get(
        "archive_sha256"
    )
    report["receiver_package"] = {
        "schema": package.get("schema"),
        "package_path": _runtime_package_path(proof),
        "archive_zip_path": archive_path,
        "archive_zip_bytes": archive_bytes,
        "archive_zip_sha256": archive_sha256,
        "runtime_consumption_proof_path": first_row.get(
            "runtime_consumption_proof_path"
        )
        or proof.get("proof_path"),
        "runtime_consumption_proof_passed": (
            proof.get("runtime_consumption_proof_passed") is True
        ),
        "receiver_contract_satisfied": (
            proof.get("receiver_contract_satisfied") is True
        ),
        "receiver_output_bytes": proof.get("receiver_output_bytes"),
        "receiver_output_sha256": proof.get("receiver_output_sha256"),
        "receiver_output_retained": proof.get("receiver_output_retained"),
        "blockers": list(proof.get("blockers") or first_row.get("blockers") or ()),
    }
    report["runtime_consumption_proof_passed"] = (
        proof.get("runtime_consumption_proof_passed") is True
    )
    report["receiver_contract_satisfied"] = (
        proof.get("receiver_contract_satisfied") is True
    )


def _runtime_package_path(proof: Mapping[str, Any]) -> str | None:
    path = proof.get("proof_path")
    if not path:
        return None
    proof_path = Path(str(path))
    try:
        return proof_path.parents[1].as_posix()
    except IndexError:
        return None


def _read_packet(path: Path) -> tuple[bytes, str, list[dict[str, Any]]]:
    blob = path.read_bytes()
    if blob.startswith((b"SNAR1", b"SNAR2")):
        return blob, "raw_snar_packet", []
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            infos = zf.infolist()
            if len(infos) != 1:
                raise ValueError("expected single-member archive.zip")
            info = infos[0]
            payload = zf.read(info.filename)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path}: expected raw SNAR packet or archive.zip") from exc
    return (
        payload,
        "archive_zip_member_packet",
        [
            {
                "filename": info.filename,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "compress_type": int(info.compress_type),
            }
        ],
    )


def _output_wire_format(value: str, *, source_schema: str) -> str:
    if value == "same":
        return "snar2" if source_schema == SNERV_ARCHIVE_SCHEMA_V2 else "snar1"
    if value in {"snar1", "snar2"}:
        return value
    raise ValueError(f"unsupported wire format: {value!r}")


def _hard_byte_ceiling_rows(
    *,
    source_packet_bytes: int,
    candidate_packet_bytes: int | None,
    ceilings: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_ceiling in ceilings:
        ceiling = int(raw_ceiling)
        rows.append(
            {
                "hard_byte_ceiling": ceiling,
                "source_packet_over_ceiling_bytes": max(
                    int(source_packet_bytes) - ceiling,
                    0,
                ),
                "candidate_packet_over_ceiling_bytes": None
                if candidate_packet_bytes is None
                else max(int(candidate_packet_bytes) - ceiling, 0),
                "candidate_packet_under_ceiling": None
                if candidate_packet_bytes is None
                else int(candidate_packet_bytes) <= ceiling,
            }
        )
    return rows


def _write_blocked_report(path: Path, report: dict[str, Any], blocker: str) -> None:
    report["blockers"] = _dedupe([*list(report.get("blockers") or ()), blocker])
    _write_report(path, report)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        required=True,
        type=Path,
        help="Raw SNAR1/SNAR2 packet or archive.zip containing member 0.bin.",
    )
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument(
        "--wire-format",
        default="snar2",
        choices=("same", "snar1", "snar2"),
        help="Output wire format. Defaults to SNAR2.",
    )
    parser.add_argument("--output-packet", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-archive-zip", type=Path)
    parser.add_argument("--output-package-dir", type=Path)
    parser.add_argument(
        "--full-video-receiver-proof",
        action="store_true",
        help="Generate runtime package and run receiver inflate proof.",
    )
    parser.add_argument(
        "--retain-receiver-output",
        action="store_true",
        help="Keep the raw receiver output after proof. Defaults to cleanup.",
    )
    parser.add_argument("--package-timeout-seconds", type=int, default=1800)
    parser.add_argument("--hard-byte-ceiling", action="append", type=int)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
