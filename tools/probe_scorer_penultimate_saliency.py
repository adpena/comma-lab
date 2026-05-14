#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit a proxy-safe scorer penultimate-feature saliency smoke manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.analysis.scorer_penultimate_saliency import (  # noqa: E402
    build_synthetic_smoke_manifest,
    validate_penultimate_saliency_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cooperative_receiver/scorer_penultimate_saliency_smoke.json"),
    )
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the manifest violates proxy/analysis-only guardrails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_synthetic_smoke_manifest(
        seed=args.seed,
        batch_size=args.batch_size,
        repo_root=args.repo_root,
    )
    violations = validate_penultimate_saliency_manifest(manifest)
    manifest["manifest_validation"] = {
        "passed": not violations,
        "violations": violations,
    }
    text = json.dumps(manifest, indent=2 if args.pretty else None, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
