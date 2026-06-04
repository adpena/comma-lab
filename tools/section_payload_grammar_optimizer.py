#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the generic section-payload grammar optimizer on named files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr101_per_tensor_grammar_solver import (  # noqa: E402
    DEFAULT_CODERS,
    CoderName,
)
from tac.packet_compiler.section_payload_grammar_optimizer import (  # noqa: E402
    build_section_payload_optimizer_queue,
    sections_from_single_member_zip_archive,
    solve_section_payload_grammar,
    spans_from_archive_section_telemetry,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _parse_section(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("section must be NAME=PATH")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("section name must be non-empty")
    parsed = Path(path).expanduser()
    if not parsed.is_file():
        raise argparse.ArgumentTypeError(f"section payload not found: {parsed}")
    return name, parsed


def _parse_zip_section(raw: str) -> dict[str, int | str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("zip section must be NAME:START:LENGTH")
    name, start_raw, length_raw = parts
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("zip section name must be non-empty")
    try:
        start = int(start_raw)
        length = int(length_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("zip section START/LENGTH must be integers") from exc
    if start < 0 or length < 0:
        raise argparse.ArgumentTypeError("zip section START/LENGTH must be non-negative")
    return {"name": name, "start": start, "length": length}


def _parse_coder(raw: str) -> CoderName:
    if raw not in DEFAULT_CODERS:
        raise argparse.ArgumentTypeError(
            f"coder must be one of {', '.join(DEFAULT_CODERS)}"
        )
    return raw  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        action="append",
        type=_parse_section,
        help="Named payload file as NAME=PATH. Repeat for multiple sections.",
    )
    parser.add_argument(
        "--zip-archive",
        type=Path,
        help=(
            "Single-member ZIP archive to inspect directly. Use --zip-section "
            "to slice the member; without spans the full member is one section."
        ),
    )
    parser.add_argument("--zip-member", help="Expected ZIP member name.")
    parser.add_argument(
        "--archive-section-telemetry-json",
        type=Path,
        help=(
            "Archive section telemetry JSON with sections[].offset/end_offset "
            "rows to use as ZIP member spans."
        ),
    )
    parser.add_argument(
        "--zip-section",
        action="append",
        type=_parse_zip_section,
        help="Named member span as NAME:START:LENGTH. Repeat for multiple sections.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queue-output", type=Path)
    parser.add_argument("--campaign-id", default="section_payload_grammar")
    parser.add_argument(
        "--coder",
        action="append",
        type=_parse_coder,
        help="Coder to test. Defaults to the full shared portfolio.",
    )
    parser.add_argument(
        "--baseline-coder",
        default="brotli",
        type=_parse_coder,
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--expected-queue-output-sha256")
    return parser.parse_args(argv)


def _read_sections(items: list[tuple[str, Path]]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name, path in items:
        if name in out:
            raise SystemExit(f"duplicate section name: {name}")
        out[name] = path.read_bytes()
    return out


def _load_section_source(args: argparse.Namespace) -> tuple[object, dict[str, Any] | None]:
    file_sections = list(args.section or [])
    if bool(file_sections) == bool(args.zip_archive):
        raise SystemExit("pass exactly one of --section or --zip-archive")
    if file_sections:
        return _read_sections(file_sections), None
    archive_path = Path(args.zip_archive).expanduser()
    if not archive_path.is_file():
        raise SystemExit(f"ZIP archive not found: {archive_path}")
    if args.archive_section_telemetry_json is not None and args.zip_section:
        raise SystemExit(
            "pass either --archive-section-telemetry-json or --zip-section, not both"
        )
    spans = args.zip_section
    telemetry_manifest: dict[str, Any] | None = None
    if args.archive_section_telemetry_json is not None:
        telemetry_path = Path(args.archive_section_telemetry_json).expanduser()
        if not telemetry_path.is_file():
            raise SystemExit(f"archive section telemetry not found: {telemetry_path}")
        telemetry_payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
        spans = spans_from_archive_section_telemetry(telemetry_payload)
        telemetry_manifest = {
            "archive_section_telemetry_path": telemetry_path.as_posix(),
            "archive_section_telemetry_schema": telemetry_payload.get("schema"),
            "archive_section_telemetry_section_count": len(spans),
        }
    sections, manifest = sections_from_single_member_zip_archive(
        archive_path.read_bytes(),
        spans=spans,
        member_name=args.zip_member,
    )
    manifest = {
        **manifest,
        "archive_path": archive_path.as_posix(),
        **(telemetry_manifest or {}),
    }
    return sections, manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    coders = tuple(args.coder or DEFAULT_CODERS)
    sections, source_manifest = _load_section_source(args)
    report = solve_section_payload_grammar(
        sections,
        coders=coders,
        brotli_quality=args.brotli_quality,
        baseline_coder=args.baseline_coder,
        campaign_id=args.campaign_id,
        source_payload_manifest=source_manifest,
    )
    try:
        report_artifact = write_json_artifact(
            args.output,
            report,
            allow_overwrite=args.allow_overwrite
            or args.expected_output_sha256 is not None,
            expected_existing_sha256=args.expected_output_sha256,
        )
        queue_artifact: dict[str, Any] | None = None
        if args.queue_output is not None:
            queue = build_section_payload_optimizer_queue(
                report,
                campaign_id=args.campaign_id,
            )
            queue_artifact = write_json_artifact(
                args.queue_output,
                queue,
                allow_overwrite=args.allow_overwrite
                or args.expected_queue_output_sha256 is not None,
                expected_existing_sha256=args.expected_queue_output_sha256,
            )
    except ArtifactWriteError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "ok": True,
                "campaign_id": report["campaign_id"],
                "section_count": report["section_count"],
                "source_kind": None
                if source_manifest is None
                else source_manifest.get("source_kind"),
                "selected_isolated_section_bytes": report["byte_accounting"][
                    "selected_isolated_section_bytes"
                ],
                "selected_saved_bytes_vs_baseline": report["byte_accounting"][
                    "selected_saved_bytes_vs_baseline"
                ],
                "output": report_artifact.path,
                "queue_output": None if queue_artifact is None else queue_artifact.path,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
