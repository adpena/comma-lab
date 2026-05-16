#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the L5 v2 C1/Z5/TT5L probe gate artifact.

This wraps measured probe observations with the recomputable probe verdict and
verdict hash expected by `l5_v2_dispatch_readiness`. It is planning-only: no
score claim, no promotion eligibility, and no dispatch authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.l5_v2_probe_disambiguator import (  # noqa: E402
    build_l5_v2_probe_gate_artifact,
    load_observations_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output_json
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = output_path.resolve()
    try:
        output_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"probe gate artifact output must stay inside repo: {output_path}"
        ) from exc
    output_str = str(output_path)
    if (
        output_str.startswith("/tmp/")
        or "/private/tmp/" in output_str
        or "/var/tmp/" in output_str
    ):
        raise ValueError(f"refusing to write L5 v2 probe gate artifact to tmp: {output_str!r}")
    observations = load_observations_json(args.input_json)
    payload = build_l5_v2_probe_gate_artifact(observations, repo_root=REPO_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verdict = payload["probe_disambiguator"]["verdict"]
    print(f"wrote {output_path}")
    print(
        "architecture_lock_allowed="
        f"{str(verdict.get('architecture_lock_allowed') is True).lower()}"
    )
    print("score_claim=false")
    return 0 if verdict.get("architecture_lock_allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
