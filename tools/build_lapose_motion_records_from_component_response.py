#!/usr/bin/env python3
"""Build LA-POSE motion records from CUDA component-response evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.lapose_motion_evidence import records_from_component_response  # noqa: E402
from tac.repo_io import json_text, read_json, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-response-json", type=Path, required=True)
    parser.add_argument("--latent-actions-json", type=Path, required=True)
    parser.add_argument("--pair-opportunities-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def _records(payload: object, key: str) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    raise SystemExit(f"expected list or object with {key}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    component_response = read_json(args.component_response_json)
    manifest = records_from_component_response(
        component_response,
        latent_actions=_records(read_json(args.latent_actions_json), "latent_actions"),
        pair_opportunities=_records(read_json(args.pair_opportunities_json), "pair_opportunities"),
        evidence_source_path=args.component_response_json.as_posix(),
        evidence_source_sha256=sha256_file(args.component_response_json),
    )
    text = json_text(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
