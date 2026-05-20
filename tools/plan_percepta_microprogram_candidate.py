#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a Pact gate plan for a Percepta-style tiny correction circuit."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.percepta_microprogram_plan import (  # noqa: E402
    ExactEvalCustody,
    MicroprogramPrototypeSpec,
    build_microprogram_plan,
    default_weight_embedded_probe_spec,
)


def _read_spec(path: Path) -> MicroprogramPrototypeSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    custody_payload = payload.pop("custody", None)
    custody = ExactEvalCustody(**custody_payload) if isinstance(custody_payload, dict) else ExactEvalCustody()
    if "opcodes" in payload:
        payload["opcodes"] = tuple(payload["opcodes"])
    return MicroprogramPrototypeSpec(**payload, custody=custody)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="JSON MicroprogramPrototypeSpec payload")
    parser.add_argument("--prototype-id", help="Override prototype id for the default spec")
    parser.add_argument("--expected-component-delta-score", type=float, default=None)
    parser.add_argument("--best-simple-edit-delta-score", type=float, default=None)
    parser.add_argument("--recompressed-archive-byte-delta", type=int, default=None)
    parser.add_argument("--encoded-program-bytes", type=int, default=None)
    parser.add_argument("--runtime-patch-bytes", type=int, default=None)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    spec = _read_spec(args.spec) if args.spec else default_weight_embedded_probe_spec()
    replacements = {}
    for key in (
        "prototype_id",
        "expected_component_delta_score",
        "best_simple_edit_delta_score",
        "recompressed_archive_byte_delta",
        "encoded_program_bytes",
        "runtime_patch_bytes",
    ):
        value = getattr(args, key)
        if value is not None:
            replacements[key] = value
    if replacements:
        spec = replace(spec, **replacements)

    payload = build_microprogram_plan(spec).as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write_json(args.output, payload)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
