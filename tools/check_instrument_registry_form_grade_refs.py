#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Warn when verdict receipts cite registry instruments without form-grade refs.

This is the ddm_vo2 two-landing guard in warn-only form.  A receipt is considered
verdict-bearing when it carries verdict/score/evidence vocabulary.  If it names
a registry instrument_id, it should also carry a local form-grade reference such
as ``form_grade_ref:<instrument_id>`` or a JSON ``form_grade_ref`` field.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / ".omx/research/ddm_vo2_20260806/INSTRUMENT_REGISTRY.jsonl"
DEFAULT_RECEIPT_ROOTS = (REPO / ".omx/research",)

VERDICT_TOKENS = (
    "verdict",
    "MEASURED",
    "DERIVED",
    "score_claim",
    "promotion_eligible",
    "d_seg",
    "d_pose",
    "archive_bytes",
)


@dataclass(frozen=True)
class MissingFormGradeRef:
    path: str
    instrument_id: str
    message: str

    def to_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "instrument_id": self.instrument_id,
            "message": self.message,
        }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_registry(path: Path = DEFAULT_REGISTRY) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        instrument_id = str(row.get("instrument_id", "")).strip()
        if instrument_id:
            ids.add(instrument_id)
    return ids


def _is_verdict_bearing(text: str) -> bool:
    return any(token in text for token in VERDICT_TOKENS)


def _has_form_grade_ref(text: str, instrument_id: str) -> bool:
    escaped = re.escape(instrument_id)
    patterns = (
        rf"form_grade_ref\s*[:=]\s*{escaped}",
        rf"instrument_form_grade_ref\s*[:=]\s*{escaped}",
        rf'"form_grade_ref"\s*:\s*"{escaped}"',
        rf'"instrument_id"\s*:\s*"{escaped}".*?"form_grade"',
    )
    return any(re.search(pattern, text, flags=re.DOTALL) for pattern in patterns)


def _iter_receipts(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file():
            out.append(path)
            continue
        if not path.exists():
            continue
        for suffix in ("*.md", "*.json", "*.jsonl"):
            out.extend(path.rglob(suffix))
    return sorted(set(out))


def scan_receipts(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    receipt_paths: list[Path] | None = None,
) -> dict[str, Any]:
    instrument_ids = load_registry(registry_path)
    receipt_paths = receipt_paths or list(DEFAULT_RECEIPT_ROOTS)
    receipts = _iter_receipts(receipt_paths)
    violations: list[MissingFormGradeRef] = []
    verdict_bearing_count = 0
    cited_receipt_count = 0

    for path in receipts:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not _is_verdict_bearing(text):
            continue
        verdict_bearing_count += 1
        cited = [instrument_id for instrument_id in instrument_ids if instrument_id in text]
        if not cited:
            continue
        cited_receipt_count += 1
        for instrument_id in cited:
            if not _has_form_grade_ref(text, instrument_id):
                violations.append(
                    MissingFormGradeRef(
                        path=_rel(path),
                        instrument_id=instrument_id,
                        message="verdict-bearing receipt cites instrument without form_grade_ref",
                    )
                )

    return {
        "schema": "instrument_registry_form_grade_ref_scan.v1",
        "registry_path": _rel(registry_path),
        "instrument_count": len(instrument_ids),
        "receipt_roots": [_rel(p) for p in receipt_paths],
        "receipt_files_scanned": len(receipts),
        "verdict_bearing_receipts": verdict_bearing_count,
        "verdict_bearing_receipts_citing_registry": cited_receipt_count,
        "missing_form_grade_ref_count": len(violations),
        "violations": [v.to_payload() for v in violations],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = scan_receipts(
        registry_path=args.registry,
        receipt_paths=args.receipts if args.receipts else None,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        count = report["missing_form_grade_ref_count"]
        print(
            f"instrument form-grade refs: {count} missing across "
            f"{report['verdict_bearing_receipts']} verdict-bearing receipts"
        )
        for violation in report["violations"][:50]:
            print(
                "WARN "
                f"{violation['path']} cites {violation['instrument_id']} "
                "without form_grade_ref"
            )
        if count > 50:
            print(f"... {count - 50} more")
    return 1 if args.strict and report["missing_form_grade_ref_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
