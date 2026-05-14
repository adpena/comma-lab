#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attach byte-closed HNeRV low-level packet custody to rate-recode profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, read_json, repo_relative, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


class HnervRateRecodePacketLinkError(ValueError):
    """Raised when a packet cannot be linked without losing custody."""


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HnervRateRecodePacketLinkError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise HnervRateRecodePacketLinkError(f"{label} must be a JSON list")
    return value


def _packet_archive_manifest(packet: dict[str, Any]) -> dict[str, Any]:
    release_surface = _require_mapping(packet.get("release_surface"), "packet.release_surface")
    files = _require_mapping(release_surface.get("files"), "packet.release_surface.files")
    archive_manifest = _require_mapping(
        files.get("archive_manifest.json"),
        "packet.release_surface.files.archive_manifest.json",
    )
    if archive_manifest.get("exists") is not True:
        raise HnervRateRecodePacketLinkError("packet archive_manifest.json does not exist")
    path = str(archive_manifest.get("path") or "")
    sha = str(archive_manifest.get("sha256") or "")
    if not path or not _repo_path(Path(path)).is_file():
        raise HnervRateRecodePacketLinkError("packet archive manifest path is missing")
    if sha256_file(_repo_path(Path(path))) != sha:
        raise HnervRateRecodePacketLinkError("packet archive manifest sha256 mismatch")
    return archive_manifest


def _accepted_attempts(candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = _require_list(candidate_result.get("attempts"), "candidate_result.attempts")
    return [
        row
        for row in attempts
        if isinstance(row, dict) and row.get("accepted_for_candidate") is True
    ]


def _match_profile_variant(
    profile: dict[str, Any],
    *,
    packet: dict[str, Any],
    candidate_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_sha = str(profile.get("source_archive_sha256") or "")
    packet_source_sha = str(packet.get("source_archive_sha256") or "")
    result_source_sha = str(candidate_result.get("source_archive_sha256") or "")
    if source_sha != packet_source_sha or source_sha != result_source_sha:
        raise HnervRateRecodePacketLinkError(
            "source archive sha256 mismatch between profile, packet, and candidate result"
        )
    if packet.get("static_packet_ready") is not True:
        raise HnervRateRecodePacketLinkError("packet is not static_packet_ready")
    archive_sha = str(packet.get("archive_sha256") or "")
    archive_bytes = packet.get("archive_bytes")
    if not _is_sha256(archive_sha) or not isinstance(archive_bytes, int):
        raise HnervRateRecodePacketLinkError("packet archive identity is incomplete")
    if archive_sha != candidate_result.get("candidate_archive_sha256"):
        raise HnervRateRecodePacketLinkError("packet archive sha256 does not match candidate result")
    if archive_bytes != candidate_result.get("candidate_archive_bytes"):
        raise HnervRateRecodePacketLinkError("packet archive bytes do not match candidate result")

    variants = _require_list(profile.get("variants"), "profile.variants")
    for attempt in _accepted_attempts(candidate_result):
        candidate_section_sha = str(attempt.get("candidate_section_sha256") or "")
        byte_delta = attempt.get("byte_delta")
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if (
                variant.get("sha256") == candidate_section_sha
                and variant.get("byte_delta_vs_source_section") == byte_delta
            ):
                return variant, attempt
    raise HnervRateRecodePacketLinkError(
        "no profile variant matched accepted candidate section sha256 and byte delta"
    )


def attach_packet_links(profile: dict[str, Any], *, packet_path: Path) -> dict[str, Any]:
    packet = _require_mapping(read_json(packet_path), f"packet {packet_path}")
    artifacts = _require_mapping(packet.get("artifacts"), "packet.artifacts")
    candidate_result_path = Path(str(artifacts.get("candidate_result") or ""))
    if not candidate_result_path.as_posix():
        raise HnervRateRecodePacketLinkError("packet missing artifacts.candidate_result")
    candidate_result = _require_mapping(
        read_json(_repo_path(candidate_result_path)),
        f"candidate_result {candidate_result_path}",
    )
    archive_manifest = _packet_archive_manifest(packet)
    variant, attempt = _match_profile_variant(
        profile,
        packet=packet,
        candidate_result=candidate_result,
    )

    packet_sha = sha256_file(_repo_path(packet_path))
    candidate_result_sha = sha256_file(_repo_path(candidate_result_path))
    archive_manifest_path = str(archive_manifest["path"])
    archive_manifest_sha = str(archive_manifest["sha256"])
    linked = {
        "variant": variant["variant"],
        "packet_path": _repo_rel(packet_path),
        "packet_sha256": packet_sha,
        "candidate_result_path": _repo_rel(candidate_result_path),
        "candidate_result_sha256": candidate_result_sha,
        "archive_manifest_path": archive_manifest_path,
        "archive_manifest_sha256": archive_manifest_sha,
        "candidate_archive_path": str(candidate_result.get("candidate_archive_path") or ""),
        "candidate_archive_sha256": str(packet["archive_sha256"]),
        "candidate_archive_bytes": int(packet["archive_bytes"]),
        "candidate_member_name": str(candidate_result.get("candidate_member_name") or ""),
        "source_archive_sha256": str(packet["source_archive_sha256"]),
        "byte_delta": int(packet["byte_delta"]),
        "section_name": str(attempt.get("section_name") or ""),
        "section_sha256": str(attempt.get("candidate_section_sha256") or ""),
        "static_packet_ready": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "submit_blockers": list(packet.get("submit_blockers") or []),
        "score_blockers": list(packet.get("score_blockers") or []),
    }
    variant.update(
        {
            "archive_manifest_path": archive_manifest_path,
            "archive_manifest_sha256": archive_manifest_sha,
            "candidate_archive_path": linked["candidate_archive_path"],
            "candidate_archive_sha256": linked["candidate_archive_sha256"],
            "candidate_archive_bytes": linked["candidate_archive_bytes"],
            "byte_closed_packet_path": linked["packet_path"],
            "byte_closed_packet_sha256": packet_sha,
            "byte_closed_candidate_packet_attached": True,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_exact_eval_dispatch": False,
        }
    )
    packets = [
        row
        for row in profile.get("byte_closed_candidate_packets", [])
        if isinstance(row, dict) and row.get("packet_path") != linked["packet_path"]
    ]
    packets.append(linked)
    profile["byte_closed_candidate_packets"] = sorted(
        packets,
        key=lambda row: (str(row.get("variant") or ""), str(row.get("packet_path") or "")),
    )
    profile["byte_closed_candidate_packet_count"] = len(profile["byte_closed_candidate_packets"])
    profile["ready_for_exact_eval_dispatch"] = False
    profile["dispatch_attempted"] = False
    return profile


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--packet", type=Path, action="append", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    profile = _require_mapping(read_json(args.profile), f"profile {args.profile}")
    for packet_path in args.packet:
        profile = attach_packet_links(profile, packet_path=packet_path)
    profile = attach_tool_run_manifest(
        profile,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.profile, *args.packet],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(profile), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
