#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan or build a fail-closed PR106 high-order context recode prototype."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_context_recode import (  # noqa: E402
    DEFAULT_CONTEXT_ORDERS,
    TARGETABLE_INNER_SECTIONS,
    build_pr106_context_recode_report,
    load_pr106_context_source_from_archive,
    load_pr106_context_source_from_payload,
    write_report_json,
    write_report_markdown,
)

DEFAULT_PR106_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip"
)


def _parse_orders(value: str) -> tuple[int, ...]:
    try:
        orders = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--orders must be comma-separated integers") from exc
    if not orders:
        raise argparse.ArgumentTypeError("--orders must include at least one context order")
    if any(order < 0 for order in orders):
        raise argparse.ArgumentTypeError("--orders cannot include negative values")
    return orders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="single-member PR106 archive.zip to profile",
    )
    source.add_argument("--payload", type=Path, help="raw PR106 payload/member bytes to profile")
    parser.add_argument(
        "--target-section",
        default="auto",
        help=(
            "target section to prototype, or auto. Targetable: "
            + ", ".join(TARGETABLE_INNER_SECTIONS)
        ),
    )
    parser.add_argument("--context-order", type=int, default=2)
    parser.add_argument("--orders", type=_parse_orders, default=DEFAULT_CONTEXT_ORDERS)
    parser.add_argument(
        "--build-prototype",
        action="store_true",
        help="emit a lossless PCR1 prototype section envelope for the selected target",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--prototype-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    archive_path = args.archive
    if archive_path is None and args.payload is None and DEFAULT_PR106_ARCHIVE.exists():
        archive_path = DEFAULT_PR106_ARCHIVE
    if archive_path is None and args.payload is None:
        parser.error("provide --archive or --payload")

    if archive_path is not None:
        source = load_pr106_context_source_from_archive(archive_path)
    else:
        source = load_pr106_context_source_from_payload(args.payload)

    result = build_pr106_context_recode_report(
        source,
        target_section=args.target_section,
        context_order=args.context_order,
        context_orders=args.orders,
        build_prototype=args.build_prototype,
    )

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "experiments/results" / f"pr106_context_recode_{timestamp}"
    json_out = args.json_out or (out_dir / "profile.json")
    md_out = args.md_out or (out_dir / "profile.md")
    prototype_out = args.prototype_out or (
        out_dir / f"prototype_order{args.context_order}.pcr1"
    )

    if result.prototype_section_bytes is not None:
        prototype_out.parent.mkdir(parents=True, exist_ok=True)
        prototype_out.write_bytes(result.prototype_section_bytes)
        candidate = result.report.get("prototype_candidate")
        if isinstance(candidate, dict):
            candidate["prototype_section_path"] = str(prototype_out)

    write_report_json(json_out, result.report)
    write_report_markdown(md_out, result.report)
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    if result.prototype_section_bytes is not None:
        print(f"wrote {prototype_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
