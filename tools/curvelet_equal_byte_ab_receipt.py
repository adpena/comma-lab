#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Govern the literal-curvelet versus Fourier equal-ZIP transfer receipt.

``match`` creates deterministic equal-byte archive copies and a custody receipt.
The caller then inflates/evaluates those exact copies.  ``finalize`` re-derives
the archive receipt, proves source/matched output-tree identity, validates the
two official measurement JSON rows, and applies the instance-scoped transfer
law.  This tool never trains, evaluates, or moves the score pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.canonical_equations.curvelet_equal_archive_transfer_20260716 import (
    EQUATION_ID,
    RECEIPT_SCHEMA,
    evaluate_curvelet_equal_archive_transfer,
)
from tac.through_r.equal_archive_budget import (
    EqualArchiveBudgetReceipt,
    MatchedArchiveReceipt,
    equalize_archive_budgets,
    verify_equal_archive_budget_receipt,
    verify_output_tree_preserved,
)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return payload


def _equal_budget_from_dict(payload: Mapping[str, Any]) -> EqualArchiveBudgetReceipt:
    try:
        left = MatchedArchiveReceipt(**dict(payload["left"]))
        right = MatchedArchiveReceipt(**dict(payload["right"]))
        return EqualArchiveBudgetReceipt(
            version=str(payload["version"]),
            padding_member=str(payload["padding_member"]),
            fixed_zip_timestamp=tuple(payload["fixed_zip_timestamp"]),
            target_archive_bytes=int(payload["target_archive_bytes"]),
            equal_archive_bytes=payload["equal_archive_bytes"],
            left=left,
            right=right,
            receipt_sha256=str(payload["receipt_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("equal-budget JSON does not match EqualArchiveBudgetReceipt") from exc


def tree_manifest_sha256(manifest: Mapping[str, str]) -> str:
    """Content-address one relative-path to file-hash output-tree manifest."""

    return _canonical_sha256({"files": dict(manifest)})


def match_archives(
    *,
    control_source: str | Path,
    treatment_source: str | Path,
    control_matched: str | Path,
    treatment_matched: str | Path,
) -> dict[str, Any]:
    """Create exact-byte copies and return their content-bound receipt."""

    Path(control_matched).parent.mkdir(parents=True, exist_ok=True)
    Path(treatment_matched).parent.mkdir(parents=True, exist_ok=True)
    return equalize_archive_budgets(
        control_source,
        treatment_source,
        control_matched,
        treatment_matched,
    ).to_dict()


def finalize_transfer(
    *,
    control_matched: str | Path,
    treatment_matched: str | Path,
    equal_budget: Mapping[str, Any],
    control_source_output: str | Path,
    control_matched_output: str | Path,
    treatment_source_output: str | Path,
    treatment_matched_output: str | Path,
    control_measurement: Mapping[str, Any],
    treatment_measurement: Mapping[str, Any],
    basis_program_sha256: str,
) -> dict[str, Any]:
    """Re-derive all gates and return a final non-pointer-authorizing receipt."""

    budget_receipt = _equal_budget_from_dict(equal_budget)
    verify_equal_archive_budget_receipt(
        control_matched,
        treatment_matched,
        budget_receipt,
    )
    control_tree = verify_output_tree_preserved(control_source_output, control_matched_output)
    treatment_tree = verify_output_tree_preserved(
        treatment_source_output,
        treatment_matched_output,
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "lawref": EQUATION_ID,
        "assembled_at_utc": _utc(),
        "basis_program_sha256": basis_program_sha256,
        "equal_budget_receipt_verified": True,
        "output_trees_preserved": True,
        "equal_budget": dict(equal_budget),
        "output_tree_custody": {
            "control_sha256": tree_manifest_sha256(control_tree),
            "treatment_sha256": tree_manifest_sha256(treatment_tree),
            "control_files": len(control_tree),
            "treatment_files": len(treatment_tree),
        },
        "measurements": {
            "control": dict(control_measurement),
            "treatment": dict(treatment_measurement),
        },
        "pointer_delta": "ZERO",
        "score_claim": False,
        "family_verdict": "OPEN",
    }
    verdict = evaluate_curvelet_equal_archive_transfer(receipt)
    receipt["verdict"] = verdict.to_dict()
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def _durable_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(Path("/tmp")) or resolved.is_relative_to(Path("/private/tmp")):
        raise ValueError("operator-facing receipt must be durable and cannot live under /tmp")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _durable_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    match = subparsers.add_parser("match", help="create exact equal-byte archive copies")
    match.add_argument("--control-source", type=Path, required=True)
    match.add_argument("--treatment-source", type=Path, required=True)
    match.add_argument("--control-matched", type=Path, required=True)
    match.add_argument("--treatment-matched", type=Path, required=True)
    match.add_argument("--receipt", type=Path, required=True)

    finalize = subparsers.add_parser("finalize", help="verify inflated outputs and measurements")
    finalize.add_argument("--control-matched", type=Path, required=True)
    finalize.add_argument("--treatment-matched", type=Path, required=True)
    finalize.add_argument("--equal-budget-receipt", type=Path, required=True)
    finalize.add_argument("--control-source-output", type=Path, required=True)
    finalize.add_argument("--control-matched-output", type=Path, required=True)
    finalize.add_argument("--treatment-source-output", type=Path, required=True)
    finalize.add_argument("--treatment-matched-output", type=Path, required=True)
    finalize.add_argument("--control-measurement", type=Path, required=True)
    finalize.add_argument("--treatment-measurement", type=Path, required=True)
    finalize.add_argument("--basis-program-sha256", required=True)
    finalize.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "match":
        payload = match_archives(
            control_source=args.control_source,
            treatment_source=args.treatment_source,
            control_matched=args.control_matched,
            treatment_matched=args.treatment_matched,
        )
    else:
        payload = finalize_transfer(
            control_matched=args.control_matched,
            treatment_matched=args.treatment_matched,
            equal_budget=_load_mapping(args.equal_budget_receipt),
            control_source_output=args.control_source_output,
            control_matched_output=args.control_matched_output,
            treatment_source_output=args.treatment_source_output,
            treatment_matched_output=args.treatment_matched_output,
            control_measurement=_load_mapping(args.control_measurement),
            treatment_measurement=_load_mapping(args.treatment_measurement),
            basis_program_sha256=args.basis_program_sha256,
        )
    _atomic_json(args.receipt, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
