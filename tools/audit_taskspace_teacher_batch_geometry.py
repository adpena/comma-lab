#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal exact target-label drift across upstream scorer batch geometries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (  # noqa: E402
    EVIDENCE_AXIS,
    FreshTeacherMaterializationError,
    atomic_write_json,
    file_identity,
    load_and_reverify_materialization_receipt,
    load_compile_ready_materialization_receipt,
    load_json_mapping,
    reverify_preflight,
)
from tac.witness_control.taskspace_teacher_batch_geometry_audit_v1 import (  # noqa: E402
    compare_label_banks,
    seal_batch_geometry_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-receipt", type=Path, required=True)
    parser.add_argument("--comparison-receipt", type=Path, action="append", default=[])
    parser.add_argument("--comparison-npz", type=Path, default=None)
    parser.add_argument("--comparison-npz-key", default="lstars")
    parser.add_argument("--comparison-npz-name", default="historical_cache")
    parser.add_argument("--comparison-npz-batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _receipt_bank(receipt: dict[str, Any]) -> np.memmap:
    target = receipt["target_labels"]
    shape = tuple(int(value) for value in target["shape"])
    if target.get("dtype") != "uint8":
        raise FreshTeacherMaterializationError("teacher target bank is not uint8")
    return np.memmap(Path(target["path"]), dtype=np.uint8, mode="r", shape=shape)


def _receipt_identity(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(receipt["target_labels"]["path"]).parent.parent
    preflight_path = output_root / "00_custody_storage_preflight.json"
    preflight = load_json_mapping(preflight_path)
    reverify_preflight(preflight)
    if preflight["preflight_sha256"] != receipt["preflight_sha256"]:
        raise FreshTeacherMaterializationError("receipt and stage-00 preflight hashes differ")
    return {
        "receipt_file": file_identity(path),
        "receipt_sha256": receipt["receipt_sha256"],
        "preflight_file": file_identity(preflight_path),
        "preflight_sha256": preflight["preflight_sha256"],
        "target_labels": receipt["target_labels"],
        "batch_size": receipt.get(
            "scorer_pair_batch_size",
            receipt.get("batch_size", preflight["batch_size"]),
        ),
        "batch_geometry_authority": receipt.get(
            "batch_geometry_authority",
            preflight.get("batch_geometry_authority", "LEGACY_RECEIPT_GEOMETRY_UNDECLARED"),
        ),
        "contest_axis_authority": receipt.get("contest_axis_authority", False),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary_path = args.primary_receipt.resolve()
    primary_receipt = load_compile_ready_materialization_receipt(primary_path)
    primary = _receipt_bank(primary_receipt)
    comparisons = []

    for path_arg in args.comparison_receipt:
        comparison_path = path_arg.resolve()
        receipt = load_and_reverify_materialization_receipt(comparison_path)
        identity = _receipt_identity(comparison_path, receipt)
        name = f"batch{identity['batch_size']}_receipt"
        comparisons.append(
            {
                "source": identity,
                "comparison": compare_label_banks(
                    primary,
                    _receipt_bank(receipt),
                    comparison_name=name,
                ),
            }
        )

    if args.comparison_npz is not None:
        if args.comparison_npz_batch_size is None or args.comparison_npz_batch_size < 1:
            raise FreshTeacherMaterializationError(
                "--comparison-npz requires a positive --comparison-npz-batch-size"
            )
        cache_path = args.comparison_npz.resolve()
        cache_identity = file_identity(cache_path)
        with np.load(cache_path, allow_pickle=False) as payload:
            if args.comparison_npz_key not in payload.files:
                raise FreshTeacherMaterializationError(
                    f"NPZ comparison lacks key {args.comparison_npz_key!r}"
                )
            comparison = compare_label_banks(
                primary,
                payload[args.comparison_npz_key],
                comparison_name=args.comparison_npz_name,
            )
        comparisons.append(
            {
                "source": {
                    "npz_file": cache_identity,
                    "npz_key": args.comparison_npz_key,
                    "batch_size": args.comparison_npz_batch_size,
                    "batch_geometry_authority": "HISTORICAL_CONTEXT_NOT_UPSTREAM_DEFAULT",
                    "contest_axis_authority": False,
                },
                "comparison": comparison,
            }
        )

    if not comparisons:
        raise FreshTeacherMaterializationError("at least one comparison bank is required")
    body = {
        "schema": "tac.taskspace_teacher_batch_geometry_audit.v1",
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "pointer_mutation_allowed": False,
        "contest_axis_authority": False,
        "primary": _receipt_identity(primary_path, primary_receipt),
        "comparisons": comparisons,
        "verdict": "PRIMARY_MATCHES_FROZEN_UPSTREAM_DEFAULT_BATCH_GEOMETRY",
        "compiler_policy": (
            "only the primary bank may feed semantic compilation; comparison banks are "
            "diagnostic context and no target/scorer byte may enter candidate payload"
        ),
    }
    sealed = seal_batch_geometry_audit(body)
    atomic_write_json(args.output.resolve(), sealed)
    return sealed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run(args)
    except (FreshTeacherMaterializationError, OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
