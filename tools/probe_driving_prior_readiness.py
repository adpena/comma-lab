#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit the 2032 driving-prior scorer-saliency readiness manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.analysis.driving_prior_readiness import (  # noqa: E402
    build_driving_prior_readiness_manifest,
    validate_readiness_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a no-download, no-GPU, no-dispatch readiness manifest for "
            "the fdfc347f 2032 driving-prior / scorer-penultimate-saliency plan."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-scorer-probe",
        action="store_true",
        help="Do not instantiate local scorer architectures or attach hook probes.",
    )
    parser.add_argument(
        "--load-scorer-weights",
        action="store_true",
        help="CPU-only optional check that loads local safetensors before hook probing.",
    )
    parser.add_argument(
        "--hash-scorer-weights",
        action="store_true",
        help="Hash local scorer safetensors in the manifest. No network is used.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero only if the emitted manifest violates proxy-safety schema.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_driving_prior_readiness_manifest(
        args.repo_root,
        probe_scorer=not args.skip_scorer_probe,
        load_weights=args.load_scorer_weights,
        hash_weights=args.hash_scorer_weights,
    )
    violations = validate_readiness_manifest(manifest)
    manifest["manifest_validation"] = {
        "passed": not violations,
        "violations": violations,
    }
    text = json.dumps(manifest, indent=2 if args.pretty else None, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
