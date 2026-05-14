#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit or validate no-score parser-section manifests for public HNeRV packets."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.hnerv_packet_sections import (  # noqa: E402
    PARSER_AUTO,
    PARSER_CHOICES,
    build_packet_section_manifest_batch,
    dumps_manifest,
    render_manifest_summary,
    validate_packet_section_manifest_batch,
)
from tac.repo_io import read_json  # noqa: E402

DEFAULT_ARCHIVES = (
    (
        "PR101",
        REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
    (
        "PR103",
        REPO_ROOT / "experiments/results/public_pr103_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
    (
        "PR106",
        REPO_ROOT / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip",
        PARSER_AUTO,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        default=None,
        help="Archive path, or LABEL=PATH. Repeatable. Defaults to local PR101/PR103/PR106 public archives.",
    )
    parser.add_argument(
        "--parser",
        choices=PARSER_CHOICES,
        default=PARSER_AUTO,
        help="Parser for all --archive inputs. Defaults to auto.",
    )
    parser.add_argument("--json-out", type=Path, help="Write canonical JSON manifest here.")
    parser.add_argument("--validate-json", type=Path, help="Validate an existing manifest JSON instead of emitting.")
    parser.add_argument("--fail-if-blocked", action="store_true", help="Exit 1 when validation blockers exist.")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_json:
        payload = read_json(args.validate_json)
        if not isinstance(payload, dict):
            raise SystemExit("manifest root must be a JSON object")
        blockers = validate_packet_section_manifest_batch(payload, repo_root=REPO_ROOT)
        if args.format == "json":
            print(dumps_manifest({"ready": not blockers, "blockers": blockers}), end="")
        else:
            print("parser-section gate: " + ("ready" if not blockers else "blocked"))
            for blocker in blockers:
                print(f"- {blocker}")
        return 1 if blockers and args.fail_if_blocked else 0

    archives = _archive_specs(args.archive, parser=args.parser)
    payload = build_packet_section_manifest_batch(archives, repo_root=REPO_ROOT)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(dumps_manifest(payload), encoding="utf-8")
    if args.format == "json":
        print(dumps_manifest(payload), end="")
    else:
        print(render_manifest_summary(payload), end="")
    blockers = payload["parser_section_gate"]["blockers"]
    return 1 if blockers and args.fail_if_blocked else 0


def _archive_specs(values: list[str] | None, *, parser: str) -> list[tuple[str, Path, str]]:
    if not values:
        return [(label, path, item_parser) for label, path, item_parser in DEFAULT_ARCHIVES]
    specs: list[tuple[str, Path, str]] = []
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
            if not label:
                raise SystemExit(f"archive label is empty: {value!r}")
        else:
            raw_path = value
            label = Path(raw_path).stem
        specs.append((label, Path(raw_path), parser))
    return specs


if __name__ == "__main__":
    raise SystemExit(main())
