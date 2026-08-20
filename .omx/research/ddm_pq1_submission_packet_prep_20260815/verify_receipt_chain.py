#!/usr/bin/env python3
"""Refuse when a tracked packet document was edited without a successor receipt.

The gen6 receipt chain (DOC_DIVERGENCE_RECEIPT*.json, append-only, frozen SSD
custody) records the sha256 of every tracked packet document each time one is
deliberately changed. Eight rv17 findings shared one genus: hand-maintained
coupling between two artifacts that must agree -- a document was edited and the
record (or the document) did not follow. This script is the machine end of that
class (rv17 R8-F1 cure): it recomputes the live shas of every document the
LATEST receipt tracks and exits non-zero on any disagreement, naming the file
and the owed cure (append the next receipt; never edit an existing one).

Usage (defaults fit the gen6 packet; all overridable for controls):

    python3 verify_receipt_chain.py \
        [--receipts DIR] [--prep DIR] [--frozen DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_RECEIPTS = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_receipts"
)
DEFAULT_FROZEN = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet/generations/gen6_rc2_composed"
)
DEFAULT_PREP = Path(__file__).resolve().parent

_RECEIPT_RE = re.compile(r"^DOC_DIVERGENCE_RECEIPT(?:_R(\d+))?\.json$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_receipt(receipts_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for path in receipts_dir.iterdir():
        match = _RECEIPT_RE.match(path.name)
        if match is None:
            continue
        # The unsuffixed receipt is the chain's oldest link (round 3).
        rank = int(match.group(1)) if match.group(1) else 3
        if best is None or rank > best[0]:
            best = (rank, path)
    return best[1] if best else None


def _check(label: str, path: Path, expected: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"{label}: MISSING on disk ({path})")
        return
    actual = _sha256(path)
    if actual != expected:
        failures.append(
            f"{label}: sha mismatch\n"
            f"  latest receipt {expected}\n"
            f"  live file      {actual}\n"
            f"  ({path})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--prep", type=Path, default=DEFAULT_PREP)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    args = parser.parse_args(argv)

    if not args.receipts.is_dir():
        print(f"FAIL: receipts dir not found: {args.receipts}", file=sys.stderr)
        return 1
    latest = _latest_receipt(args.receipts)
    if latest is None:
        print(f"FAIL: no DOC_DIVERGENCE_RECEIPT*.json in {args.receipts}", file=sys.stderr)
        return 1
    receipt = json.loads(latest.read_text())

    failures: list[str] = []
    checked = 0
    for name, entry in receipt.get("diverged_files", {}).items():
        if "repo_final_sha256" in entry:
            _check(f"{name} (repo)", args.prep / name, entry["repo_final_sha256"], failures)
            checked += 1
        if "frozen_gen6_sha256" in entry:
            _check(f"{name} (frozen)", args.frozen / name, entry["frozen_gen6_sha256"], failures)
            checked += 1
    for name, entry in receipt.get("repo_only_docs", {}).items():
        if "repo_final_sha256" in entry:
            _check(f"{name} (repo)", args.prep / name, entry["repo_final_sha256"], failures)
            checked += 1

    if checked == 0:
        print(f"FAIL: latest receipt {latest.name} tracks zero documents", file=sys.stderr)
        return 1
    if failures:
        print(f"latest receipt: {latest.name}", file=sys.stderr)
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            "A tracked document changed without a successor receipt (rv17 R8-F1\n"
            "class). Cure: append DOC_DIVERGENCE_RECEIPT_R<next>.json recording\n"
            "the new shas; never edit an existing receipt.",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: {checked} tracked document shas match the latest receipt ({latest.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
