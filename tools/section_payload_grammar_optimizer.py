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
    solve_section_payload_grammar,
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
        required=True,
        help="Named payload file as NAME=PATH. Repeat for multiple sections.",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    coders = tuple(args.coder or DEFAULT_CODERS)
    report = solve_section_payload_grammar(
        _read_sections(args.section),
        coders=coders,
        brotli_quality=args.brotli_quality,
        baseline_coder=args.baseline_coder,
        campaign_id=args.campaign_id,
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
