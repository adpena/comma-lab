#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a typed fail-closed null-seed candidate spec.

The output is a lowering target for future runtime adapters. It does not
rewrite archives, emit a submission packet, or claim score movement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.procedural_codebook_generator import (  # noqa: E402
    build_null_seed_candidate_spec,
    render_null_seed_candidate_spec_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a typed null-seed candidate spec from a "
            "null_space_seed_replacement_plan_v1 JSON file."
        )
    )
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--archive-zip", type=Path)
    parser.add_argument("--inner-member-name")
    parser.add_argument("--seed-hex")
    parser.add_argument("--generator-kind", default="pcg64")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"{args.plan_json} did not contain a JSON object")
    seed_material = bytes.fromhex(args.seed_hex) if args.seed_hex else None
    spec = build_null_seed_candidate_spec(
        plan,
        candidate_id=args.candidate_id,
        candidate_rank=args.candidate_rank,
        archive_zip_path=args.archive_zip,
        inner_member_name=args.inner_member_name,
        seed_material=seed_material,
        generator_kind=args.generator_kind,
    )
    spec["input_paths"] = {
        "plan_json": str(args.plan_json),
        "archive_zip_override": str(args.archive_zip) if args.archive_zip else None,
        "inner_member_name_override": args.inner_member_name,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_null_seed_candidate_spec_markdown(spec),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": spec["schema"],
                "spec_id": spec["spec_id"],
                "verdict": spec["verdict"],
                "direct_replacement_ready": spec["direct_replacement_ready"],
                "runtime_adapter_required": spec["runtime_adapter_required"],
                "score_claim": spec["score_claim"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
