#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing SIREN substrate first-anchor readiness gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.substrates.siren_readiness import audit_siren_substrate_readiness  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_siren_substrate_readiness",
        description=(
            "Fail-closed local readiness audit for lane_substrate_siren_20260512. "
            "This never dispatches, never claims score, and never promotes proxy output."
        ),
    )
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p.add_argument("--output", type=Path, help="Optional path to write the JSON manifest.")
    p.add_argument(
        "--fail-if-not-ready",
        action="store_true",
        help="Exit nonzero when local SIREN trainer/recipe/runtime surfaces are incomplete.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = audit_siren_substrate_readiness(args.repo_root)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload["summary"])
        print(f"lane_id: {payload['lane_id']}")
        print(f"local_contract_ready: {payload['local_contract_ready']}")
        print("ready_for_remote_dispatch: false")
        if payload["local_blockers"]:
            print("local blockers:")
            for blocker in payload["local_blockers"]:
                print(f"  - {blocker}")
        print("dispatch blockers:")
        for blocker in payload["dispatch_blockers"]:
            print(f"  - {blocker}")

    if args.fail_if_not_ready and not payload["local_contract_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
