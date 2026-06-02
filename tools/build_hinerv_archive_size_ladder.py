#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a measured false-authority HiNeRV archive-size ladder."""

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

from comma_lab.storage_tiers import DEFAULT_RESERVE_FREE_GB  # noqa: E402
from tac.analysis.hinerv_archive_size_ladder import (  # noqa: E402
    HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
    build_hinerv_archive_size_ladder,
    render_hinerv_archive_size_ladder_markdown,
)
from tac.repo_io import write_json  # noqa: E402

DEFAULT_STORAGE_EXPECTED_BYTES = 512 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", default=None, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--num-pairs", default=600, type=int)
    parser.add_argument("--row-id", action="append", default=None)
    parser.add_argument("--decoder-codec", default="int8_mixed")
    parser.add_argument("--emit-receiver-proof", action="store_true")
    parser.add_argument("--retain-receiver-proof-output", action="store_true")
    parser.add_argument("--emit-decoder-weight-waterfill-plan", action="store_true")
    parser.add_argument("--decoder-weight-saliency-json", default=None, type=Path)
    parser.add_argument("--decoder-weight-waterfill-action-bits", default="0,2,4,8,16,32")
    parser.add_argument(
        "--allow-local-output-dir",
        action="store_true",
        help=(
            "Permit archive ladder artifacts on local disk. Default refuses local "
            "outputs so bulky rebuildable archives land on the SSD tier."
        ),
    )
    parser.add_argument(
        "--storage-expected-bytes",
        default=DEFAULT_STORAGE_EXPECTED_BYTES,
        type=int,
        help="Expected output bytes for the storage preflight.",
    )
    parser.add_argument(
        "--storage-reserve-free-gb",
        default=DEFAULT_RESERVE_FREE_GB,
        type=float,
        help="Free-space reserve required after expected output bytes.",
    )
    args = parser.parse_args(argv)

    report = build_hinerv_archive_size_ladder(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        num_pairs=int(args.num_pairs),
        row_ids=args.row_id,
        decoder_codec=str(args.decoder_codec),
        emit_receiver_proof=bool(args.emit_receiver_proof),
        retain_receiver_proof_output=bool(args.retain_receiver_proof_output),
        allow_local_output_dir=bool(args.allow_local_output_dir),
        storage_expected_bytes=int(args.storage_expected_bytes),
        storage_reserve_free_gb=float(args.storage_reserve_free_gb),
        emit_decoder_weight_waterfill_plan=bool(
            args.emit_decoder_weight_waterfill_plan
        ),
        decoder_weight_saliency_json=args.decoder_weight_saliency_json,
        decoder_weight_waterfill_action_bits=_parse_action_bits(
            args.decoder_weight_waterfill_action_bits
        ),
    )
    output = args.output_json.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    write_json(output, report)
    if args.output_md is not None:
        md_output = args.output_md.expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
        md_output.write_text(
            render_hinerv_archive_size_ladder_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": HINERV_ARCHIVE_SIZE_LADDER_SCHEMA,
        "report_path": report.get("report_path"),
        "output_dir": report["output_dir"],
        "row_count": report["row_count"],
        "archive_bytes": {
            row["row_id"]: row["archive_bytes"] for row in report["archive_rows"]
        },
        "emit_decoder_weight_waterfill_plan": report[
            "emit_decoder_weight_waterfill_plan"
        ],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


def _parse_action_bits(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(value).split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
