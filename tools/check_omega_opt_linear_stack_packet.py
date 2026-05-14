#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate an Omega-OPT linear-stack packet manifest.

This checker is intentionally local and scoreless: it only verifies that a
linear-stack packet manifest stays fail-closed unless the exact A++ archive,
runtime/inflate, and CUDA auth-eval anchor fields are present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.omega_opt_linear_stack_packet import (  # noqa: E402
    linear_stack_packet_status,
    validate_linear_stack_packet_manifest,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest)
    findings = validate_linear_stack_packet_manifest(manifest)
    status = linear_stack_packet_status(manifest)

    if args.format == "json":
        print(json.dumps({
            "ok": not findings,
            "finding_count": len(findings),
            "findings": findings,
            "promotion_status": status,
        }, indent=2, sort_keys=True))
    elif findings:
        print(f"OMEGA-OPT LINEAR-STACK PACKET FINDINGS ({len(findings)}):")
        for finding in findings:
            print(f"  - {finding}")
    else:
        print("Omega-OPT linear-stack packet manifest: PASS")

    if findings and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
