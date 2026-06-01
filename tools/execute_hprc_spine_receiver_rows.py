#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute receiver proofs for selected HPRC spine bounded-runner rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.spine_receiver_execution import (  # noqa: E402
    HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA,
    SpineReceiverRuntimeOverride,
    execute_spine_receiver_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--row-id", action="append", default=[])
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--runtime-override",
        action="append",
        default=[],
        metavar="FAMILY=INFLATE_SH",
        help="Override runtime inflate.sh for a family. Repeatable.",
    )
    parser.add_argument(
        "--expected-raw-bytes-override",
        action="append",
        default=[],
        metavar="FAMILY_OR_ROW=BYTES",
        help="Testing/recovery override for expected raw bytes. Repeatable.",
    )
    parser.add_argument(
        "--output-contract-override",
        action="append",
        default=[],
        metavar="FAMILY_OR_ROW=raw_file|png_tree",
        help="Override expected receiver output contract. Repeatable.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--allow-large-output", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    runtime_overrides = [
        SpineReceiverRuntimeOverride(family=family, inflate_sh=path)
        for family, path in _parse_path_map(args.runtime_override, repo_root=repo_root).items()
    ]
    report = execute_spine_receiver_rows(
        runner_plan_path=args.runner_plan,
        output_dir=args.output_dir,
        repo_root=repo_root,
        row_ids=list(args.row_id),
        max_rows=args.max_rows,
        runtime_overrides=runtime_overrides,
        timeout_seconds=float(args.timeout_seconds),
        max_output_bytes=int(args.max_output_bytes),
        allow_large_output=bool(args.allow_large_output),
        expected_raw_bytes_overrides=_parse_int_map(args.expected_raw_bytes_override),
        output_contract_overrides=_parse_str_map(args.output_contract_override),
        keep_work_dir=bool(args.keep_work_dir),
        allow_overwrite=bool(args.force),
    )
    print(
        json.dumps(
            {
                "schema": HPRC_SPINE_RECEIVER_EXECUTION_REPORT_SCHEMA,
                "report_path": report["report_path"],
                "deduped_execution_row_count": report["deduped_execution_row_count"],
                "receiver_proof_passed_count": report["receiver_proof_passed_count"],
                "receiver_proof_blocked_count": report["receiver_proof_blocked_count"],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0 if int(report["receiver_proof_blocked_count"]) == 0 else 1


def _parse_path_map(raw_items: list[str], *, repo_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in raw_items:
        family, value = _split_map_item(raw)
        path = Path(value).expanduser()
        out[family] = path if path.is_absolute() else (repo_root / path).resolve(strict=False)
    return out


def _parse_int_map(raw_items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in raw_items:
        key, value = _split_map_item(raw)
        out[key] = int(value)
    return out


def _parse_str_map(raw_items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in raw_items:
        key, value = _split_map_item(raw)
        if value not in {"raw_file", "png_tree"}:
            raise ValueError(f"unsupported output contract: {value!r}")
        out[key] = value
    return out


def _split_map_item(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected KEY=VALUE: {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"expected nonempty KEY=VALUE: {raw!r}")
    return key, value


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"execute_hprc_spine_receiver_rows failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
