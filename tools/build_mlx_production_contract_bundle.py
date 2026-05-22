#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a strict non-authoritative MLX production-contract bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac.local_acceleration.mlx_production_contract import (
    BUNDLE_FAIL_VERDICT,
    build_mlx_scorer_production_contract_bundle_manifest,
    load_json_object,
    write_production_contract_manifest,
)
from tac.optimization.scorer_response_dataset import build_mlx_production_contract_gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        action="append",
        default=[],
        type=Path,
        help="Strict MLX production contract JSON path. May repeat.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dataset",
        type=Path,
        help=(
            "Optional scorer-response dataset. When provided, the output bundle "
            "is failed unless every MLX scorer-response row is covered by a "
            "strict child contract."
        ),
    )
    return parser


def _mlx_rows_from_dataset(dataset: dict) -> list[dict]:
    rows = dataset.get("rows")
    if not isinstance(rows, list):
        raise ValueError("dataset rows[] missing")
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            row.get("family") == "mlx_scorer_response"
            or row.get("source_schema") == "mlx_scorer_response.v1"
        )
    ]


def _attach_dataset_coverage_gate(
    manifest: dict,
    *,
    dataset: dict,
) -> dict:
    mlx_rows = _mlx_rows_from_dataset(dataset)
    gate = build_mlx_production_contract_gate(manifest, rows=mlx_rows)
    manifest["dataset_coverage_gate"] = gate
    manifest["summary"]["dataset_mlx_row_count"] = len(mlx_rows)
    manifest["summary"]["dataset_matched_row_count"] = gate["summary"].get(
        "matched_row_count"
    )
    if gate.get("status") != "strict_pass":
        blockers = list(manifest.get("blockers") or [])
        blockers.append("dataset_mlx_row_coverage_gate_not_strict_pass")
        blockers.extend(str(item) for item in gate.get("blockers") or [])
        manifest["blockers"] = blockers
        manifest["passed"] = False
        manifest["verdict"] = BUNDLE_FAIL_VERDICT
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    contracts = [load_json_object(path) for path in args.contract]
    manifest = build_mlx_scorer_production_contract_bundle_manifest(
        contracts,
        run_id=args.run_id,
        producer="tools.build_mlx_production_contract_bundle",
    )
    if args.dataset is not None:
        manifest = _attach_dataset_coverage_gate(
            manifest,
            dataset=load_json_object(args.dataset),
        )
    write_production_contract_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": manifest["passed"],
                "verdict": manifest["verdict"],
                "contract_count": manifest["summary"]["contract_count"],
                "strict_contract_count": manifest["summary"]["strict_contract_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
