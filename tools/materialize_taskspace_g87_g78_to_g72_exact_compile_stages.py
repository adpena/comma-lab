#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize the complete exact G78-to-G72 proposal universe."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tac.witness_control.taskspace_g87_g78_to_g72_exact_compile_adapter_v1 import (
    G87ExactCompileAdapterError,
    materialize_complete_g72_proposal_stages,
    open_g87_g78_to_g72_compile_input,
)

CONFIG_SCHEMA: Final = "tac.taskspace_g87_g78_to_g72_exact_compile_materializer_config.v1"


@dataclass(frozen=True, slots=True)
class G87MaterializerConfigV1:
    aggregate_receipt_path: Path
    aggregate_file_sha256: str
    aggregate_self_sha256: str
    output_root: Path


def load_config(path: Path) -> G87MaterializerConfigV1:
    """Load the exact closed-key typed materializer config."""

    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G87ExactCompileAdapterError("G87 materializer config cannot be read") from exc
    if type(value) is not dict or set(value) != {
        "aggregate_file_sha256",
        "aggregate_receipt_path",
        "aggregate_self_sha256",
        "output_root",
        "schema",
    }:
        raise G87ExactCompileAdapterError("G87 materializer config key set differs")
    if value["schema"] != CONFIG_SCHEMA:
        raise G87ExactCompileAdapterError("G87 materializer config schema differs")
    for field in (
        "aggregate_receipt_path",
        "aggregate_file_sha256",
        "aggregate_self_sha256",
        "output_root",
    ):
        if type(value[field]) is not str or not value[field]:
            raise G87ExactCompileAdapterError(f"G87 materializer config {field} is not a nonempty string")
    return G87MaterializerConfigV1(
        aggregate_receipt_path=Path(value["aggregate_receipt_path"]),
        aggregate_file_sha256=value["aggregate_file_sha256"],
        aggregate_self_sha256=value["aggregate_self_sha256"],
        output_root=Path(value["output_root"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    compile_input = open_g87_g78_to_g72_compile_input(
        config.aggregate_receipt_path,
        expected_file_sha256=config.aggregate_file_sha256,
        expected_self_sha256=config.aggregate_self_sha256,
    )
    receipt_path = config.output_root / "aggregate_receipt.json"
    if args.status:
        if not receipt_path.is_file():
            print(
                json.dumps(
                    {
                        "status": "not_materialized",
                        "output_root": str(config.output_root),
                        "g87_compile_input_receipt_sha256": (compile_input.receipt["compile_input_receipt_sha256"]),
                    },
                    sort_keys=True,
                )
            )
            return 1
        print(receipt_path.read_text(encoding="utf-8").strip())
        return 0
    receipt_path, receipt = materialize_complete_g72_proposal_stages(
        compile_input,
        output_root=config.output_root,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "receipt_path": str(receipt_path),
                "materialization_receipt_sha256": receipt["materialization_receipt_sha256"],
                "proposal_count": receipt["population"]["proposal_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
