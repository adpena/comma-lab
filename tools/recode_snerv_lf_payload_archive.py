#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Losslessly recode the LF payload inside a full SNeRV SNAR1 packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_lf_payload_archive_recode import (  # noqa: E402
    DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    SCHEMA,
    build_snerv_lf_payload_archive_recode,
    render_snerv_lf_payload_archive_recode_markdown,
)
from tac.repo_io import write_json  # noqa: E402


def _default_json_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_lf_payload_archive_recode_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-json",
        type=Path,
        default=None,
        help=(
            "Optional snerv_lf_payload_codec_sweep.v1 report. When --packet or "
            "--mode are omitted, this supplies source.path and selected mode."
        ),
    )
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--output-packet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--frame-proof-max-output-bytes",
        type=int,
        default=DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
        help=(
            "Run streaming receiver frame equality only when estimated output "
            "bytes are <= this cap. LF exactness and unchanged-section proof "
            "always run."
        ),
    )
    parser.add_argument(
        "--force-frame-proof",
        action="store_true",
        help="Run streaming receiver frame equality even over the output-byte cap.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting output packet/report paths.",
    )
    args = parser.parse_args(argv)

    sweep = _load_sweep(args.sweep_json)
    packet_path = _resolve_packet_path(args.packet, sweep)
    mode = _resolve_mode(args.mode, sweep)
    output_json = (
        args.output_json.expanduser().resolve(strict=False)
        if args.output_json is not None
        else _default_json_path()
    )
    output_packet = args.output_packet.expanduser().resolve(strict=False)
    _check_output(output_packet, allow_overwrite=bool(args.allow_overwrite))
    _check_output(output_json, allow_overwrite=bool(args.allow_overwrite))
    if args.output_md is not None:
        _check_output(
            args.output_md.expanduser().resolve(strict=False),
            allow_overwrite=bool(args.allow_overwrite),
        )

    source_bytes = packet_path.read_bytes()
    report, candidate_packet = build_snerv_lf_payload_archive_recode(
        source_bytes,
        mode=mode,
        source_packet_path=packet_path.as_posix(),
        frame_proof_max_output_bytes=int(args.frame_proof_max_output_bytes),
        force_frame_proof=bool(args.force_frame_proof),
    )
    output_packet.parent.mkdir(parents=True, exist_ok=True)
    output_packet.write_bytes(candidate_packet)
    packet_sha = hashlib.sha256(candidate_packet).hexdigest()
    report["candidate_packet"]["path"] = output_packet.as_posix()
    report["candidate_packet"]["file_bytes"] = len(candidate_packet)
    report["candidate_packet"]["file_sha256"] = packet_sha
    report["candidate_packet"]["file_matches_report"] = (
        packet_sha == report["candidate_packet"]["sha256"]
        and len(candidate_packet) == report["candidate_packet"]["bytes"]
    )
    if sweep:
        report["source_sweep"] = _sweep_ref(sweep, args.sweep_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output_json.as_posix()
    write_json(output_json, report)
    if args.output_md is not None:
        output_md = args.output_md.expanduser().resolve(strict=False)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = output_md.as_posix()
        output_md.write_text(
            render_snerv_lf_payload_archive_recode_markdown(report),
            encoding="utf-8",
        )

    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _load_sweep(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve(strict=False)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "snerv_lf_payload_codec_sweep.v1":
        raise SystemExit(
            f"--sweep-json must be snerv_lf_payload_codec_sweep.v1, got {payload.get('schema')!r}"
        )
    payload["_resolved_path"] = resolved.as_posix()
    return payload


def _resolve_packet_path(path: Path | None, sweep: dict[str, Any]) -> Path:
    if path is not None:
        return path.expanduser().resolve(strict=False)
    source = sweep.get("source") if isinstance(sweep.get("source"), dict) else {}
    if source.get("kind") != "snar1_packet" or not source.get("path"):
        raise SystemExit(
            "--packet is required unless --sweep-json came from a SNAR1 packet"
        )
    return Path(str(source["path"])).expanduser().resolve(strict=False)


def _resolve_mode(mode: str | None, sweep: dict[str, Any]) -> str:
    if mode:
        return str(mode)
    selected = (
        sweep.get("selected_rate_only_row")
        if isinstance(sweep.get("selected_rate_only_row"), dict)
        else {}
    )
    selected_mode = str(selected.get("mode") or "")
    if not selected_mode:
        raise SystemExit("--mode is required unless --sweep-json has selected mode")
    return selected_mode


def _check_output(path: Path, *, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {path}")


def _sweep_ref(sweep: dict[str, Any], path: Path | None) -> dict[str, Any]:
    text = json.dumps(
        {k: v for k, v in sweep.items() if k != "_resolved_path"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": sweep.get("_resolved_path") or (path.as_posix() if path else None),
        "schema": sweep.get("schema"),
        "selected_mode": (
            sweep.get("selected_rate_only_row", {}).get("mode")
            if isinstance(sweep.get("selected_rate_only_row"), dict)
            else None
        ),
        "sha256_canonical_json": hashlib.sha256(text).hexdigest(),
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "report_path": report.get("report_path"),
        "candidate_packet_path": report.get("candidate_packet", {}).get("path"),
        "mode": report.get("mode"),
        "source_packet_bytes": report.get("source_packet", {}).get("bytes"),
        "candidate_packet_bytes": report.get("candidate_packet", {}).get("bytes"),
        "packet_byte_delta": report.get("packet_byte_delta"),
        "lf_byte_delta": report.get("lf_payload", {}).get("byte_delta"),
        "lf_planes_exact_equal": report.get("lf_planes_exact_equal"),
        "receiver_frame_equality_status": report.get(
            "receiver_frame_equality_proof",
            {},
        ).get("status"),
        "receiver_contract_satisfied": report.get("receiver_contract_satisfied"),
        "score_claim": report.get("score_claim"),
        "ready_for_exact_eval_dispatch": report.get("ready_for_exact_eval_dispatch"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
